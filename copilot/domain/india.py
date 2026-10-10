"""India-specific normalisation: cities, CTC in LPA, notice periods, experience bands.

All functions are pure and deterministic so they can run at ingestion time and in tests.
"""

from __future__ import annotations

import re

from copilot.domain.models import WorkMode

CITY_ALIASES: dict[str, str] = {
    "bangalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "blr": "Bengaluru",
    "gurgaon": "Gurugram",
    "gurugram": "Gurugram",
    "noida": "Noida",
    "greater noida": "Noida",
    "new delhi": "Delhi",
    "delhi": "Delhi",
    "delhi ncr": "Delhi NCR",
    "ncr": "Delhi NCR",
    "bombay": "Mumbai",
    "mumbai": "Mumbai",
    "navi mumbai": "Mumbai",
    "madras": "Chennai",
    "chennai": "Chennai",
    "hyderabad": "Hyderabad",
    "secunderabad": "Hyderabad",
    "pune": "Pune",
    "kolkata": "Kolkata",
    "calcutta": "Kolkata",
    "ahmedabad": "Ahmedabad",
    "kochi": "Kochi",
    "cochin": "Kochi",
    "trivandrum": "Thiruvananthapuram",
    "thiruvananthapuram": "Thiruvananthapuram",
    "jaipur": "Jaipur",
    "chandigarh": "Chandigarh",
    "indore": "Indore",
    "coimbatore": "Coimbatore",
}

_INDIA_MARKERS = ("india", "remote - india", "india - remote", "remote india")


def normalise_cities(location: str) -> list[str]:
    """Extract canonical Indian city names from a free-text location string.

    >>> normalise_cities("Bangalore / Gurgaon (Hybrid)")
    ['Bengaluru', 'Gurugram']
    """
    text = location.lower()
    found: list[str] = []
    # Longest aliases first so "navi mumbai" wins over "mumbai".
    for alias in sorted(CITY_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", text):
            city = CITY_ALIASES[alias]
            if city not in found:
                found.append(city)
            text = re.sub(rf"\b{re.escape(alias)}\b", " ", text)
    return found


def is_india_location(location: str, location_filter: list[str] | None = None) -> bool:
    """True when a posting's location is in India (or matches the configured filter)."""
    text = location.lower()
    if normalise_cities(location) or any(m in text for m in _INDIA_MARKERS):
        return True
    if location_filter:
        return any(f.lower() in text for f in location_filter)
    return False


_LPA_RANGE = re.compile(
    r"(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(?:l|lpa|lakhs?)?\s*(?:-|–|to)\s*"
    r"(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(?:l|lpa|lakhs?|lacs?)\b",
    re.IGNORECASE,
)
_LPA_SINGLE = re.compile(
    r"(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(?:lpa|lakhs?|lacs?)\b(?:\s*per\s*annum)?",
    re.IGNORECASE,
)
_CRORE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:cr|crore)s?\b", re.IGNORECASE)


def parse_ctc_lpa(text: str) -> tuple[float | None, float | None]:
    """Parse a CTC range in lakhs per annum.

    Handles "12-18 LPA", "₹12L – ₹18L", "15 lakhs", "1.2 Cr". Returns (min, max).
    """
    if m := _LPA_RANGE.search(text):
        lo, hi = float(m.group(1)), float(m.group(2))
        return (min(lo, hi), max(lo, hi))
    if m := _CRORE.search(text):
        value = float(m.group(1)) * 100.0
        return (value, value)
    if m := _LPA_SINGLE.search(text):
        value = float(m.group(1))
        return (value, value)
    return (None, None)


_IMMEDIATE = re.compile(r"\bimmediate(?:ly)?\s+(?:joiner|joining|start)s?\b", re.IGNORECASE)
_NOTICE_DAYS = re.compile(
    r"(?:notice\s*(?:period)?\s*(?:of|:)?\s*(?:max(?:imum)?|up\s*to|<=|less\s*than)?\s*"
    r"(\d{1,3})\s*days?)|(?:(\d{1,3})\s*days?\s*(?:notice|np)\b)",
    re.IGNORECASE,
)
_NOTICE_MONTHS = re.compile(
    r"(?:notice\s*(?:period)?\s*(?:of|:)?\s*(?:max(?:imum)?|up\s*to)?\s*(\d)\s*months?)"
    r"|(?:(\d)\s*months?\s*notice)",
    re.IGNORECASE,
)


def parse_notice_period_days(text: str) -> int | None:
    """Maximum notice period a JD accepts, in days. 'Immediate joiner' -> 0."""
    if _IMMEDIATE.search(text):
        return 0
    if m := _NOTICE_DAYS.search(text):
        return int(m.group(1) or m.group(2))
    if m := _NOTICE_MONTHS.search(text):
        return int(m.group(1) or m.group(2)) * 30
    return None


_EXP_RANGE = re.compile(
    r"(\d{1,2}(?:\.\d)?)\s*(?:\+)?\s*(?:-|–|to)\s*(\d{1,2}(?:\.\d)?)\s*(?:\+)?\s*(?:yrs?|years?)",
    re.IGNORECASE,
)
_EXP_MIN = re.compile(r"(\d{1,2}(?:\.\d)?)\s*\+\s*(?:yrs?|years?)", re.IGNORECASE)


def parse_experience_band(text: str) -> tuple[float | None, float | None]:
    """Parse "3-5 yrs", "3 to 5 years", "5+ years" into (min, max)."""
    if m := _EXP_RANGE.search(text):
        lo, hi = float(m.group(1)), float(m.group(2))
        return (min(lo, hi), max(lo, hi))
    if m := _EXP_MIN.search(text):
        return (float(m.group(1)), None)
    return (None, None)


def infer_work_mode(text: str) -> WorkMode:
    lowered = text.lower()
    if "hybrid" in lowered:
        return WorkMode.HYBRID
    if "remote" in lowered or "work from home" in lowered or "wfh" in lowered:
        return WorkMode.REMOTE_INDIA if "india" in lowered else WorkMode.REMOTE_GLOBAL
    if "onsite" in lowered or "on-site" in lowered or "in office" in lowered:
        return WorkMode.ONSITE
    return WorkMode.UNKNOWN
