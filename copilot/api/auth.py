"""Verify the HS256 JWT that the Next.js app (Auth.js) forwards as a Bearer token.

Stdlib only, so it is testable without PyJWT. Rejects 'none' and any non-HS256 alg.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any


class AuthError(Exception):
    pass


def _b64url_decode(part: str) -> bytes:
    return base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def sign_token(claims: dict[str, Any], secret: str) -> str:
    """Create an HS256 JWT. Used by tests and the dev login helper."""
    header = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64url_encode(json.dumps(claims, separators=(",", ":")).encode())
    signing_input = f"{header}.{payload}".encode()
    signature = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    return f"{header}.{payload}.{_b64url_encode(signature)}"


def verify_token(token: str, secret: str, audience: str | None = None, leeway_s: int = 30) -> dict[str, Any]:
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
        header = json.loads(_b64url_decode(header_b64))
        claims = json.loads(_b64url_decode(payload_b64))
    except (ValueError, json.JSONDecodeError) as exc:
        raise AuthError("malformed token") from exc
    if header.get("alg") != "HS256":
        raise AuthError("unsupported algorithm")
    expected = hmac.new(secret.encode(), f"{header_b64}.{payload_b64}".encode(), hashlib.sha256).digest()
    if not hmac.compare_digest(expected, _b64url_decode(sig_b64)):
        raise AuthError("bad signature")
    now = time.time()
    if "exp" in claims and now > float(claims["exp"]) + leeway_s:
        raise AuthError("token expired")
    if audience and claims.get("aud") != audience:
        raise AuthError("wrong audience")
    if not claims.get("sub"):
        raise AuthError("token has no subject")
    return claims
