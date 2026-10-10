"""Composition root: the only module that knows which adapter implements which port.

Swap infrastructure by changing configuration, not code:
  * LLM_PROVIDER_PROFILE=fake EMBEDDING_PROFILE=hashing VECTOR_STORE=memory
    runs the whole pipeline offline with no services.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from copilot.adapters.calendar import IcsCalendarClient
from copilot.adapters.documents import DocxResumeRenderer, MarkdownResumeRenderer
from copilot.adapters.embeddings import build_embedder
from copilot.adapters.http import UrllibHttpClient
from copilot.adapters.jobsources import AdzunaJobSource, GreenhouseJobSource, LeverJobSource
from copilot.adapters.llm import build_llm_client
from copilot.adapters.prompts import FilePromptStore
from copilot.adapters.repositories import InMemoryApplicationRepository
from copilot.adapters.tracing import InMemoryTracer, LangfuseTracer, NoopTracer
from copilot.adapters.vectorstores import InMemoryVectorStore, QdrantVectorStore
from copilot.config.settings import Secrets, Settings
from copilot.graph.gates import ReviewGate
from copilot.graph.nodes import PipelineNodes, PipelineServices
from copilot.ports.documents import DocumentRenderer
from copilot.ports.embeddings import Embedder
from copilot.ports.http import HttpClient
from copilot.ports.job_source import JobSource
from copilot.ports.llm import LLMClient
from copilot.ports.tracing import Tracer
from copilot.ports.vector_store import VectorStore
from copilot.services.applications import ApplicationService
from copilot.services.claim_verifier import ClaimVerifier
from copilot.services.experiments import ABRouter
from copilot.services.fit_scorer import FitScorer
from copilot.services.indexing import CollectionNames, IndexingService
from copilot.services.interview_prep import InterviewPrepService
from copilot.services.job_search import JobSearchService
from copilot.services.query_builder import QueryBuilder
from copilot.services.resume_parser import ResumeParser
from copilot.services.retrieval import Retriever
from copilot.services.tailor import TailorService


def build_tracer(settings: Settings) -> Tracer:
    s = settings.secrets
    if settings.tracing_enabled and not (
        Secrets.is_dummy(s.langfuse_public_key) or Secrets.is_dummy(s.langfuse_secret_key)
    ):
        assert s.langfuse_public_key and s.langfuse_secret_key
        return LangfuseTracer(
            settings.langfuse_host,
            s.langfuse_public_key.get_secret_value(),
            s.langfuse_secret_key.get_secret_value(),
        )
    return InMemoryTracer() if settings.env == "dev" else NoopTracer()


def build_vector_store(settings: Settings) -> VectorStore:
    if settings.vector_store == "memory":
        return InMemoryVectorStore()
    key = settings.secrets.qdrant_api_key
    return QdrantVectorStore(
        settings.qdrant_url,
        api_key=None if Secrets.is_dummy(key) else key.get_secret_value(),  # type: ignore[union-attr]
    )


def build_job_sources(settings: Settings, http: HttpClient) -> list[JobSource]:
    cfg = settings.companies
    sources: list[JobSource] = []
    s = settings.secrets
    if not (Secrets.is_dummy(s.adzuna_app_id) or Secrets.is_dummy(s.adzuna_app_key)):
        assert s.adzuna_app_id and s.adzuna_app_key
        sources.append(
            AdzunaJobSource(
                http,
                s.adzuna_app_id.get_secret_value(),
                s.adzuna_app_key.get_secret_value(),
                country=settings.adzuna_country,
            )
        )
    if greenhouse := cfg.active("greenhouse"):
        sources.append(
            GreenhouseJobSource(
                http, greenhouse, cfg.location_filter, cfg.adapters["greenhouse"].endpoint
            )
        )
    if lever := cfg.active("lever"):
        sources.append(LeverJobSource(http, lever, cfg.location_filter, cfg.adapters["lever"].endpoint))
    return sources


def build_renderer() -> DocumentRenderer:
    has_docx = importlib.util.find_spec("docx") is not None
    return DocxResumeRenderer() if has_docx else MarkdownResumeRenderer()


@dataclass
class Container:
    settings: Settings
    tracer: Tracer
    llm: LLMClient
    embedder: Embedder
    store: VectorStore
    services: PipelineServices
    calendar: IcsCalendarClient

    def nodes(self, gate: ReviewGate) -> PipelineNodes:
        return PipelineNodes(self.services, gate)


def build_container(
    settings: Settings,
    *,
    llm: LLMClient | None = None,
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
    job_sources: list[JobSource] | None = None,
    tracer: Tracer | None = None,
) -> Container:
    """Wire everything. Keyword overrides let tests inject fakes for any port."""
    tracer = tracer or build_tracer(settings)
    llm = llm or build_llm_client(settings.llm, tracer, settings.secrets)
    embedder = embedder or build_embedder(settings.llm)
    store = store or build_vector_store(settings)
    prompts = FilePromptStore(settings.prompts_dir)
    names = CollectionNames(settings.app.collections, settings.llm.active_embedding.collection_suffix)
    indexing = IndexingService(store, embedder, names)
    retriever = Retriever(store, embedder, names, tracer, settings.app.retrieval.min_score)
    sources = job_sources if job_sources is not None else build_job_sources(settings, UrllibHttpClient())

    services = PipelineServices(
        resume_parser=ResumeParser(llm, prompts),
        query_builder=QueryBuilder(),
        job_search=JobSearchService(sources, tracer),
        indexing=indexing,
        retriever=retriever,
        fit_scorer=FitScorer(llm=llm, prompts=prompts),
        tailor=TailorService(llm, prompts),
        verifier=ClaimVerifier({c.name for c in settings.companies.companies}),
        renderer=build_renderer(),
        interview_prep=InterviewPrepService(retriever, llm, prompts),
        ab_router=ABRouter(settings.app.experiments),
        applications=ApplicationService(InMemoryApplicationRepository()),
        tracer=tracer,
        pipeline=settings.app.pipeline,
        output_dir=settings.output_dir,
    )
    return Container(
        settings=settings,
        tracer=tracer,
        llm=llm,
        embedder=embedder,
        store=store,
        services=services,
        calendar=IcsCalendarClient(settings.output_dir / "calendar"),
    )
