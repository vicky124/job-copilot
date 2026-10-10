"""docgen-mcp: renders an APPROVED tailored resume to DOCX (Markdown fallback).

Write tool: only bound to graph nodes that run after human_review approval.
Idempotent: the same profile + draft produce the same file name.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from copilot.container import build_renderer
from copilot.domain.models import CandidateProfile, TailoredResume
from copilot.ports.documents import DocumentRenderer
from mcp_servers.common import create_server, run


def render_resume_tool(
    renderer: DocumentRenderer,
    output_dir: Path,
    profile_json: dict[str, Any],
    tailored_json: dict[str, Any],
) -> dict[str, str]:
    profile = CandidateProfile.model_validate(profile_json)
    tailored = TailoredResume.model_validate(tailored_json)
    doc = renderer.render_resume(profile, tailored, output_dir)
    return {"path": str(doc.path), "format": doc.format}


def build_server() -> Any:
    renderer = build_renderer()
    output_dir = Path(os.environ.get("APP_OUTPUT_DIR", "outputs")) / "docgen"
    server = create_server("docgen-mcp", "Render approved, tailored resumes to DOCX.")

    @server.tool()
    def render_resume(profile_json: dict[str, Any], tailored_json: dict[str, Any]) -> dict[str, str]:
        """Render an approved TailoredResume for a CandidateProfile. Returns the file path."""
        return render_resume_tool(renderer, output_dir, profile_json, tailored_json)

    return server


if __name__ == "__main__":  # pragma: no cover
    run(build_server())
