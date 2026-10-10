"""PII redaction applied to every span attribute before export."""

from __future__ import annotations

import re
from typing import Any

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Indian mobiles (+91 optional, 10 digits starting 6-9) and generic long digit runs.
_PHONE = re.compile(r"(?:\+?91[\s-]?)?(?<!\d)[6-9]\d{4}[\s-]?\d{5}(?!\d)")
_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_AADHAAR = re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")


def redact_pii(value: Any) -> Any:
    """Mask emails, Indian phone numbers, PAN and Aadhaar in strings (recursively)."""
    if isinstance(value, str):
        out = _EMAIL.sub("[EMAIL]", value)
        out = _AADHAAR.sub("[AADHAAR]", out)
        out = _PHONE.sub("[PHONE]", out)
        return _PAN.sub("[PAN]", out)
    if isinstance(value, dict):
        return {k: redact_pii(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(redact_pii(v) for v in value)
    return value
