"""Ports: the interfaces the services depend on (Dependency Inversion).

Each port is a small ``typing.Protocol`` (Interface Segregation). Adapters in
``copilot.adapters`` implement them; tests use fakes from ``tests.fakes``.
"""

from copilot.ports.calendar import CalendarClient, CalendarEvent
from copilot.ports.documents import DocumentRenderer, RenderedDocument
from copilot.ports.embeddings import Embedder
from copilot.ports.http import HttpClient, HttpError
from copilot.ports.job_source import JobSource
from copilot.ports.llm import LLMClient, LLMError, LLMMessage, LLMResponse
from copilot.ports.prompts import Prompt, PromptNotFoundError, PromptStore
from copilot.ports.repositories import ApplicationRecord, ApplicationRepository
from copilot.ports.tracing import Span, Tracer
from copilot.ports.vector_store import SearchHit, VectorStore

__all__ = [
    "ApplicationRecord",
    "ApplicationRepository",
    "CalendarClient",
    "CalendarEvent",
    "DocumentRenderer",
    "Embedder",
    "HttpClient",
    "HttpError",
    "JobSource",
    "LLMClient",
    "LLMError",
    "LLMMessage",
    "LLMResponse",
    "Prompt",
    "PromptNotFoundError",
    "PromptStore",
    "RenderedDocument",
    "SearchHit",
    "Span",
    "Tracer",
    "VectorStore",
]
