from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from copilot.domain.models import CandidateProfile, TailoredResume

DocFormat = Literal["docx", "md"]


@dataclass(frozen=True, slots=True)
class RenderedDocument:
    path: Path
    format: DocFormat


@runtime_checkable
class DocumentRenderer(Protocol):
    def render_resume(
        self,
        profile: CandidateProfile,
        tailored: TailoredResume,
        output_dir: Path,
    ) -> RenderedDocument: ...
