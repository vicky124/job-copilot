"""Shared normalisation applied to every source's raw posting (Single Responsibility:
sources fetch and map fields; this module derives India fields and skills)."""

from __future__ import annotations

import html
import re

from copilot.domain.india import (
    infer_work_mode,
    normalise_cities,
    parse_ctc_lpa,
    parse_experience_band,
    parse_notice_period_days,
)
from copilot.domain.models import JobPosting, WorkMode
from copilot.domain.skills import extract_skills

_TAG = re.compile(r"<[^>]+>")
_BLOCK_TAG = re.compile(r"</?(?:p|div|li|ul|ol|br|h\d)[^>]*>", re.IGNORECASE)
_SPACES = re.compile(r"[ \t]+")


def strip_html(text: str) -> str:
    """HTML (possibly entity-escaped, as Greenhouse returns it) to plain text."""
    unescaped = html.unescape(html.unescape(text or ""))
    with_breaks = _BLOCK_TAG.sub("\n", unescaped)
    plain = _TAG.sub("", with_breaks)
    lines = [_SPACES.sub(" ", line).strip() for line in plain.splitlines()]
    return "\n".join(line for line in lines if line)


def enrich_posting(posting: JobPosting) -> JobPosting:
    """Fill derived fields that the source did not provide."""
    text = f"{posting.title}\n{posting.location}\n{posting.description}"
    ctc_min, ctc_max = (posting.ctc_min_lpa, posting.ctc_max_lpa)
    if ctc_min is None and ctc_max is None:
        ctc_min, ctc_max = parse_ctc_lpa(text)
    exp_min, exp_max = (posting.exp_min, posting.exp_max)
    if exp_min is None and exp_max is None:
        exp_min, exp_max = parse_experience_band(text)
    updates = {
        "locations": posting.locations or normalise_cities(posting.location),
        "ctc_min_lpa": ctc_min,
        "ctc_max_lpa": ctc_max,
        "exp_min": exp_min,
        "exp_max": exp_max,
        "notice_period_days": posting.notice_period_days
        if posting.notice_period_days is not None
        else parse_notice_period_days(text),
        "work_mode": posting.work_mode
        if posting.work_mode is not WorkMode.UNKNOWN
        else infer_work_mode(f"{posting.location} {posting.description[:1500]}"),
        "required_skills": posting.required_skills or extract_skills(posting.description),
    }
    return posting.model_copy(update=updates)
