"""jobboard-mcp: read-only job search across Adzuna India, Greenhouse, Lever and pasted JDs.

Run (stdio):            python -m mcp_servers.jobboard.server
Run (streamable HTTP):  MCP_TRANSPORT=streamable-http python -m mcp_servers.jobboard.server
"""

from __future__ import annotations

import os
from typing import Any

from copilot.adapters.http import UrllibHttpClient
from copilot.adapters.tracing import NoopTracer
from copilot.config.settings import load_settings
from copilot.container import build_job_sources
from copilot.services.job_search import JobSearchService
from mcp_servers.common import create_server, run
from mcp_servers.jobboard.tools import JobBoardTools


def build_server() -> Any:
    settings = load_settings(os.environ.get("APP_ROOT", "."))
    sources = build_job_sources(settings, UrllibHttpClient())
    tools = JobBoardTools(JobSearchService(sources, NoopTracer()), [s.name for s in sources])
    server = create_server(
        "jobboard-mcp",
        "Search Indian job listings. Read-only. Results include CTC (LPA), notice period, "
        "work mode and city. Show 'Jobs by Adzuna' attribution next to Adzuna results.",
    )

    @server.tool()
    def search_jobs(
        roles: list[str],
        locations: list[str] | None = None,
        work_modes: list[str] | None = None,
        experience_years: float | None = None,
        min_ctc_lpa: float | None = None,
        limit: int = 25,
    ) -> dict[str, Any]:
        """Search jobs by role, Indian city, work mode (onsite|hybrid|remote-india),
        experience and minimum CTC in lakhs per annum."""
        return tools.search_jobs(roles, locations, work_modes, experience_years, min_ctc_lpa, limit)

    @server.tool()
    def parse_job(text: str, url: str = "") -> dict[str, Any]:
        """Parse a job description the user pasted (Naukri, LinkedIn, Instahyre...)."""
        return tools.parse_job(text, url)

    @server.tool()
    def list_sources() -> list[str]:
        """Job sources enabled in this deployment."""
        return tools.list_sources()

    return server


if __name__ == "__main__":  # pragma: no cover
    run(build_server())
