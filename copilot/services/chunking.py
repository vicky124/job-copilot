"""Section-aware chunking for job descriptions.

Requirements stay one bullet per chunk so fit scoring can pair each requirement
with evidence; other sections are split by size with overlap.
"""

from __future__ import annotations

import re

from copilot.domain.models import Chunk, JobPosting

SECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "requirements": re.compile(
        r"^(requirements?|qualifications?|what (you|we).{0,20}(need|looking for)|"
        r"must[- ]haves?|you should apply if|skills required|who you are)\b",
        re.IGNORECASE,
    ),
    "nice_to_have": re.compile(
        r"^(nice[- ]to[- ]haves?|preferred|bonus|good to have)\b", re.IGNORECASE
    ),
    "responsibilities": re.compile(
        r"^(responsibilities|what you.{0,10}(do|own)|role|key responsibilities|the role)\b",
        re.IGNORECASE,
    ),
    "about_company": re.compile(r"^(about (us|the company|[a-z]+)|who we are)\b", re.IGNORECASE),
}

_BULLET = re.compile(r"^\s*(?:[-*•▪◦]|\d+[.)])\s+")
_BOILERPLATE = re.compile(
    r"(equal opportunity employer|we celebrate diversity|reasonable accommodation|"
    r"does not discriminate|recruitment (agency|consultanc))",
    re.IGNORECASE,
)


def split_sections(text: str) -> list[tuple[str, list[str]]]:
    """Group lines under the last recognised heading. Default section: 'general'."""
    sections: list[tuple[str, list[str]]] = [("general", [])]
    for raw in text.splitlines():
        line = raw.strip()
        if not line or _BOILERPLATE.search(line):
            continue
        heading = line.rstrip(":").strip()
        matched = next(
            (name for name, pat in SECTION_PATTERNS.items() if len(heading) < 60 and pat.match(heading)),
            None,
        )
        if matched:
            sections.append((matched, []))
            continue
        sections[-1][1].append(line)
    return [(name, lines) for name, lines in sections if lines]


def _windows(words: list[str], size: int, overlap: int) -> list[str]:
    if size <= overlap:
        raise ValueError("chunk size must exceed overlap")
    step = size - overlap
    return [" ".join(words[i : i + size]) for i in range(0, max(len(words) - overlap, 1), step)]


def chunk_job(posting: JobPosting, max_words: int = 300, overlap: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    base_meta = {
        "job_id": posting.job_id,
        "company": posting.company,
        "title": posting.title,
        "source": posting.source,
        "work_mode": posting.work_mode.value,
    }
    for section, lines in split_sections(posting.description):
        if section in {"requirements", "nice_to_have"}:
            pieces = [_BULLET.sub("", line) for line in lines if len(line) > 3]
        else:
            pieces = _windows(" ".join(lines).split(), max_words, overlap)
        for piece in pieces:
            if piece.strip():
                chunks.append(
                    Chunk(
                        id=f"{posting.job_id}#{len(chunks)}",
                        text=piece.strip(),
                        metadata={**base_meta, "section": section},
                    )
                )
    return chunks
