"""Plain Markdown resume. No dependencies; used in dev and as an ATS-safe fallback."""

from __future__ import annotations

import re
from pathlib import Path

from copilot.domain.models import CandidateProfile, TailoredResume
from copilot.ports.documents import RenderedDocument

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(*parts: str) -> str:
    return "_".join(_SAFE.sub("-", p).strip("-") for p in parts if p)[:120]


def resume_sections(profile: CandidateProfile, tailored: TailoredResume) -> list[tuple[str, list[str]]]:
    """Ordered (heading, lines) shared by every renderer so formats stay consistent."""
    contact = " | ".join(p for p in (profile.email, profile.phone, profile.location) if p)
    roles = [
        f"{r.title}, {r.company}"
        + (f" ({r.start:%b %Y} – {r.end:%b %Y})" if r.start and r.end else "")
        + (f" ({r.start:%b %Y} – Present)" if r.start and not r.end else "")
        for r in profile.experience
    ]
    return [
        (profile.full_name, [contact] if contact else []),
        ("Summary", [tailored.summary]),
        ("Key achievements", tailored.bullets),
        ("Skills", [", ".join(sorted(profile.skills))] if profile.skills else []),
        ("Experience", roles),
        ("Education", profile.education),
    ]


class MarkdownResumeRenderer:
    def render_resume(
        self, profile: CandidateProfile, tailored: TailoredResume, output_dir: Path
    ) -> RenderedDocument:
        output_dir.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for i, (heading, body) in enumerate(resume_sections(profile, tailored)):
            lines.append(f"# {heading}" if i == 0 else f"\n## {heading}")
            bullet = "" if i < 2 else "- "
            lines.extend(f"{bullet}{line}" for line in body)
        path = output_dir / f"{safe_filename(profile.full_name, tailored.job_id)}.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return RenderedDocument(path=path, format="md")
