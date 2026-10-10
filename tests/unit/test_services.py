import json
import unittest

from copilot.adapters.embeddings import HashingEmbedder
from copilot.adapters.llm import FakeLLMClient
from copilot.adapters.llm.offline import offline_llm
from copilot.adapters.prompts import FilePromptStore
from copilot.adapters.repositories import InMemoryApplicationRepository
from copilot.adapters.tracing import InMemoryTracer
from copilot.adapters.vectorstores import InMemoryVectorStore
from copilot.config.settings import CollectionsConfig, ExperimentConfig
from copilot.domain.models import (
    ApplicationStatus,
    Decision,
    EvidencePair,
    InterviewQuestion,
    ReviewDecision,
    RoundType,
    TailoredResume,
    WorkMode,
)
from copilot.ports.repositories import InvalidTransitionError
from copilot.services.applications import ApplicationNotFoundError, ApplicationService
from copilot.services.chunking import chunk_job, split_sections
from copilot.services.claim_verifier import ClaimVerifier
from copilot.services.experiments import ABRouter
from copilot.services.fit_scorer import FitScorer, ctc_fit, experience_fit, notice_fit
from copilot.services.indexing import CollectionNames, IndexingService
from copilot.services.interview_prep import BASELINE_QUESTIONS, InterviewPrepService
from copilot.services.job_search import JobSearchService, dedupe
from copilot.services.query_builder import QueryBuilder, SearchPreferences
from copilot.services.resume_parser import ResumeParser
from copilot.services.retrieval import Retriever
from copilot.services.tailor import TailorService
from tests.fakes import GOOD_TAILOR, ROOT, StaticJobSource, make_job, make_profile, make_query

PROMPTS = FilePromptStore(ROOT / "prompts")


def _draft(**overrides) -> TailoredResume:
    data = {"job_id": "test:j1", **GOOD_TAILOR}
    data.update(overrides)
    return TailoredResume(**data)


class ResumeParserTest(unittest.TestCase):
    def test_parses_profile_with_small_tier(self) -> None:
        reply = make_profile().model_dump(mode="json")
        llm = FakeLLMClient(default=json.dumps(reply))
        profile = ResumeParser(llm, PROMPTS).parse_text("Asha Rao. Backend engineer. " * 5)
        self.assertEqual(profile.full_name, "Asha Rao")
        self.assertEqual(llm.calls[0].tier, "small")
        self.assertEqual(llm.calls[0].tags["node"], "parse_resume")

    def test_offline_llm_parses_any_resume_to_sample_profile(self) -> None:
        profile = ResumeParser(offline_llm(), PROMPTS).parse_text("Someone else. Data analyst. " * 5)
        self.assertEqual(profile.full_name, "Asha Rao")
        self.assertIn("FastAPI", profile.skills)

    def test_rejects_tiny_input(self) -> None:
        with self.assertRaises(ValueError):
            ResumeParser(FakeLLMClient(), PROMPTS).parse_text("hi")


class QueryBuilderTest(unittest.TestCase):
    def test_uses_preferences(self) -> None:
        q = QueryBuilder().build(
            make_profile(),
            SearchPreferences(roles=["sre"], locations=["Pune"], work_modes=[WorkMode.HYBRID]),
        )
        self.assertEqual((q.roles, q.locations, q.work_modes), (["sre"], ["Pune"], [WorkMode.HYBRID]))
        self.assertEqual(q.ctc_min_lpa, 24)  # falls back to expected CTC
        self.assertEqual(q.exp_years, 4.2)

    def test_derives_role_without_seniority(self) -> None:
        q = QueryBuilder().build(make_profile(), SearchPreferences())
        self.assertEqual(q.roles, ["Software Engineer"])

    def test_no_roles_no_experience(self) -> None:
        with self.assertRaises(ValueError):
            QueryBuilder().build(make_profile(experience=[]), SearchPreferences())


class JobSearchTest(unittest.TestCase):
    def test_dedupes_by_id_and_content(self) -> None:
        a = make_job(source="greenhouse", external_id="1")
        same_jd = make_job(source="adzuna", external_id="2")
        other = make_job(source="lever", external_id="3", title="SRE", description="different")
        unique, removed = dedupe([a, a, same_jd, other])
        self.assertEqual([p.job_id for p in unique], ["greenhouse:1", "lever:3"])
        self.assertEqual(removed, 2)

    def test_isolates_failing_source(self) -> None:
        good = StaticJobSource([make_job()], name="good")
        bad = StaticJobSource([], name="bad", error=RuntimeError("down"))
        result = JobSearchService([good, bad], InMemoryTracer()).search(make_query())
        self.assertEqual(len(result.postings), 1)
        self.assertEqual(result.errors, {"bad": "down"})

    def test_respects_limit(self) -> None:
        jobs = [make_job(external_id=str(i), description=f"jd {i}") for i in range(5)]
        result = JobSearchService([StaticJobSource(jobs)], InMemoryTracer()).search(make_query(limit=2))
        self.assertEqual(len(result.postings), 2)


class ChunkingTest(unittest.TestCase):
    def test_sections_and_boilerplate(self) -> None:
        text = "About us\nWe pay.\nRequirements:\n- Python\n- SQL\nWe are an equal opportunity employer."
        sections = dict(split_sections(text))
        self.assertEqual(sections["requirements"], ["- Python", "- SQL"])
        self.assertNotIn("equal opportunity", json.dumps(sections).lower())

    def test_requirements_one_bullet_per_chunk(self) -> None:
        chunks = chunk_job(make_job())
        reqs = [c for c in chunks if c.metadata["section"] == "requirements"]
        self.assertEqual(reqs[0].text, "3-5 yrs Python backend")
        self.assertTrue(all(c.metadata["job_id"] == "test:j1" for c in chunks))
        self.assertEqual(len({c.id for c in chunks}), len(chunks))


class IndexingAndRetrievalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryVectorStore()
        self.names = CollectionNames(CollectionsConfig(), "v1_hash256")
        self.indexing = IndexingService(self.store, HashingEmbedder(256), self.names)
        self.retriever = Retriever(self.store, HashingEmbedder(256), self.names, InMemoryTracer())

    def test_collection_names_carry_suffix(self) -> None:
        self.assertEqual(self.names.job_descriptions, "job_descriptions_v1_hash256")

    def test_reindex_replaces_job_chunks(self) -> None:
        job = make_job()
        first = self.indexing.index_job(job)
        self.indexing.index_job(job)
        self.assertEqual(self.store.count(self.names.job_descriptions), first)

    def test_requirements_and_evidence(self) -> None:
        self.indexing.index_job(make_job())
        self.indexing.index_profile("u1", make_profile())
        reqs = self.retriever.job_requirements("test:j1")
        self.assertTrue(reqs)
        self.assertTrue(all(r.metadata["section"] == "requirements" for r in reqs))
        pairs = self.retriever.evidence_for("u1", reqs)
        self.assertEqual(len(pairs), len(reqs))
        self.assertTrue(any(p.evidence for p in pairs))

    def test_evidence_is_scoped_to_user(self) -> None:
        self.indexing.index_job(make_job())
        self.indexing.index_profile("other", make_profile())
        pairs = self.retriever.evidence_for("u1", self.retriever.job_requirements("test:j1"))
        self.assertTrue(all(p.evidence is None for p in pairs))

    def test_question_retrieval(self) -> None:
        q = InterviewQuestion(question="Explain Redis eviction", round_type=RoundType.TECHNICAL)
        self.indexing.index_questions([q])
        hits = self.retriever.interview_questions("redis caching")
        self.assertEqual(hits[0].metadata["round_type"], "technical")


class FitScorerTest(unittest.TestCase):
    def test_component_functions(self) -> None:
        self.assertEqual(experience_fit(4, 3, 5), 1.0)
        self.assertEqual(experience_fit(1, 3, 5), 0.5)
        self.assertEqual(notice_fit(90, 30), 0.0)
        self.assertEqual(notice_fit(None, 30), 1.0)
        self.assertEqual(ctc_fit(30, 20), 0.5)

    def test_strong_candidate_scores_high(self) -> None:
        evidence = [EvidencePair(requirement="Python", evidence="Built Python", evidence_chunk_id="e1")]
        score = FitScorer().score(make_profile(), make_job(), evidence)
        self.assertGreaterEqual(score.score, 85)
        self.assertEqual(score.gaps, [])
        self.assertEqual(score.evidence_ids, ["e1"])
        self.assertIn("Matches 5 of 5", score.rationale)

    def test_gaps_lower_score(self) -> None:
        job = make_job(required_skills=["java", "spring", "kafka", "python"])
        score = FitScorer().score(make_profile(), job, [])
        self.assertEqual(score.gaps, ["java", "kafka", "spring"])
        self.assertLess(score.score, 60)

    def test_llm_writes_rationale_only(self) -> None:
        llm = FakeLLMClient(default="Strong fit.")
        score = FitScorer(llm=llm, prompts=PROMPTS).score(make_profile(), make_job(), [])
        self.assertEqual(score.rationale, "Strong fit.")
        self.assertEqual(llm.calls[0].tier, "medium")


class ClaimVerifierTest(unittest.TestCase):
    def test_faithful_draft_passes(self) -> None:
        self.assertEqual(ClaimVerifier().verify(make_profile(), _draft()), [])

    def test_flags_new_skill_number_employer_credential(self) -> None:
        draft = _draft(
            summary="Led Kafka at Stripe",
            bullets=["Saved 75% cost", "AWS Certified architect", "Holds an MBA"],
        )
        reasons = " ".join(v.reason for v in ClaimVerifier({"Stripe"}).verify(make_profile(), draft))
        for expected in ("kafka", "75", "stripe", "certified", "mba"):
            self.assertIn(expected, reasons.lower())

    def test_own_employer_allowed(self) -> None:
        draft = _draft(bullets=["Built FastAPI services at PayQuick"])
        self.assertEqual(ClaimVerifier({"PayQuick"}).verify(make_profile(), draft), [])


class TailorServiceTest(unittest.TestCase):
    def test_tailors_with_strong_tier_and_redacts_pii(self) -> None:
        llm = FakeLLMClient(default=json.dumps(GOOD_TAILOR))
        out = TailorService(llm, PROMPTS).tailor(make_profile(), make_job(), [], prompt_variant="variant-b")
        self.assertEqual(out.bullets, GOOD_TAILOR["bullets"])
        self.assertEqual(out.prompt_variant, "variant-b")
        call = llm.calls[0]
        self.assertEqual(call.tier, "strong")
        prompt = "\n".join(m.content for m in call.messages)
        self.assertNotIn("asha.rao@example.com", prompt)
        self.assertNotIn("98765", prompt)
        self.assertIn("ATS optimisation", prompt)  # variant-b prompt

    def test_feedback_included_on_retry(self) -> None:
        from copilot.domain.models import ClaimViolation

        llm = FakeLLMClient(default=json.dumps(GOOD_TAILOR))
        TailorService(llm, PROMPTS).tailor(
            make_profile(), make_job(), [], violations=[ClaimViolation(bullet="x", reason="new skills")]
        )
        self.assertIn("new skills", llm.calls[0].messages[1].content)

    def test_apply_edits(self) -> None:
        edited = TailorService.apply_edits(
            _draft(),
            ReviewDecision(job_id="test:j1", decision=Decision.EDIT, edited_bullets=["Mine"]),
        )
        self.assertEqual(edited.bullets, ["Mine"])
        self.assertEqual(edited.summary, GOOD_TAILOR["summary"])
        self.assertIn("edited by user", edited.change_log)


class InterviewPrepTest(unittest.TestCase):
    def test_pack_includes_retrieved_and_baseline(self) -> None:
        store, names = InMemoryVectorStore(), CollectionNames(CollectionsConfig(), "s")
        indexing = IndexingService(store, HashingEmbedder(64), names)
        indexing.index_questions(
            [InterviewQuestion(question="How does Redis eviction work?", round_type=RoundType.TECHNICAL, skill="redis")]
        )
        retriever = Retriever(store, HashingEmbedder(64), names, InMemoryTracer())
        answers = {"answers": {"How does Redis eviction work?": "S/T/A/R"}}
        service = InterviewPrepService(retriever, FakeLLMClient(default=json.dumps(answers)), PROMPTS)
        pack = service.build_pack(make_profile(), make_job())
        texts = [q.question for q in pack.questions]
        self.assertIn("How does Redis eviction work?", texts)
        self.assertTrue(all(b.question in texts for b in BASELINE_QUESTIONS))
        self.assertEqual(pack.answers["How does Redis eviction work?"], "S/T/A/R")


class ABRouterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.router = ABRouter({"exp": ExperimentConfig(variants={"production": 0.8, "variant-b": 0.2})})

    def test_sticky_per_user(self) -> None:
        self.assertEqual(self.router.variant("exp", "u1"), self.router.variant("exp", "u1"))

    def test_split_roughly_matches_weights(self) -> None:
        share = sum(self.router.variant("exp", f"user-{i}") == "variant-b" for i in range(2000)) / 2000
        self.assertAlmostEqual(share, 0.2, delta=0.04)

    def test_unknown_experiment(self) -> None:
        self.assertEqual(self.router.variant("missing", "u1"), "production")


class ApplicationServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ApplicationService(InMemoryApplicationRepository())

    def test_lifecycle(self) -> None:
        self.service.record_draft("u1", "j1", _draft())
        record = self.service.approve("u1", "j1", "/tmp/r.docx")
        self.assertEqual(record.status, ApplicationStatus.APPROVED)
        record = self.service.set_status(record.id, ApplicationStatus.APPLIED, user_id="u1")
        self.assertEqual(record.status, ApplicationStatus.APPLIED)

    def test_save_is_idempotent(self) -> None:
        self.assertEqual(self.service.save_job("u1", "j1").id, self.service.save_job("u1", "j1").id)

    def test_invalid_transition(self) -> None:
        record = self.service.save_job("u1", "j1")
        with self.assertRaises(InvalidTransitionError):
            self.service.set_status(record.id, ApplicationStatus.OFFER)

    def test_other_users_record_is_not_found(self) -> None:
        record = self.service.save_job("u1", "j1")
        with self.assertRaises(ApplicationNotFoundError):
            self.service.set_status(record.id, ApplicationStatus.REJECTED, user_id="u2")

    def test_repository_returns_copies(self) -> None:
        record = self.service.save_job("u1", "j1")
        record.status = ApplicationStatus.OFFER  # mutate the copy only
        self.assertEqual(self.service.list("u1")[0].status, ApplicationStatus.SAVED)


if __name__ == "__main__":
    unittest.main()
