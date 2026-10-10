"""Resume file -> text -> CandidateProfile (LLM structured output, small tier)."""

from __future__ import annotations

from pathlib import Path

from copilot.domain.models import CandidateProfile
from copilot.ports.llm import LLMClient
from copilot.ports.prompts import PromptStore
from copilot.services.structured import complete_structured


class UnsupportedResumeFormatError(ValueError):
    pass


def extract_text(path: Path) -> str:
    """Plain text from .pdf, .docx, .txt or .md."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".docx":
        from docx import Document  # noqa: PLC0415 - optional dependency

        return "\n".join(p.text for p in Document(str(path)).paragraphs if p.text.strip())
    if suffix == ".pdf":
        from pypdf import PdfReader  # noqa: PLC0415 - optional dependency

        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    raise UnsupportedResumeFormatError(f"unsupported resume format: {suffix}")


class ResumeParser:
    PROMPT = "parse_resume"

    def __init__(self, llm: LLMClient, prompts: PromptStore) -> None:
        self._llm = llm
        self._prompts = prompts

    def parse_text(self, resume_text: str) -> CandidateProfile:
        if len(resume_text.strip()) < 50:
            raise ValueError("resume text is too short to parse")
        prompt = self._prompts.get(self.PROMPT)
        system, user = prompt.render(resume_text=resume_text[:20_000])
        return complete_structured(
            self._llm,
            "small",
            system,
            user,
            CandidateProfile,
            tags={"node": "parse_resume", "prompt_version": prompt.version},
        )

    def parse_file(self, path: Path) -> CandidateProfile:
        return self.parse_text(extract_text(path))
