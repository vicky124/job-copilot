"""Jobs the user pastes (Naukri, LinkedIn, Instahyre...). We never scrape those
portals; the user supplies the JD text and, optionally, its URL."""

from __future__ import annotations

import hashlib
import re

from copilot.adapters.jobsources.enrich import enrich_posting
from copilot.domain.models import JobPosting

_HEADER = re.compile(r"^(title|role|position|company|location)\s*[:\-]\s*(.+)$", re.IGNORECASE)
_PORTALS = {
    "naukri.com": "naukri",
    "linkedin.com": "linkedin",
    "instahyre.com": "instahyre",
    "cutshort.io": "cutshort",
    "foundit.in": "foundit",
}


class PastedJobParser:
    """Turns raw JD text into a JobPosting. Not a JobSource: it does not search."""

    def parse(self, text: str, url: str = "", title: str = "", company: str = "") -> JobPosting:
        if not text.strip():
            raise ValueError("pasted job description is empty")
        fields: dict[str, str] = {}
        for line in text.splitlines()[:15]:
            if m := _HEADER.match(line.strip()):
                fields.setdefault(m.group(1).lower(), m.group(2).strip())
        resolved_title = title or fields.get("title") or fields.get("role") or fields.get(
            "position"
        ) or text.strip().splitlines()[0][:120]
        digest = hashlib.sha256(f"{url}|{text}".encode()).hexdigest()[:16]
        return enrich_posting(
            JobPosting(
                source=self._portal(url),
                external_id=digest,
                title=resolved_title,
                company=company or fields.get("company", "Unknown"),
                location=fields.get("location", ""),
                url=url,
                description=text.strip(),
            )
        )

    @staticmethod
    def _portal(url: str) -> str:
        return next((name for host, name in _PORTALS.items() if host in url), "pasted")
