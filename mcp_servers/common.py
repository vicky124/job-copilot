"""Shared MCP plumbing.

Compatible with the MCP Python SDK 1.x (``FastMCP``) and 2.x (``MCPServer``).
Servers authenticate callers with a shared bearer token (MCP_SHARED_BEARER_TOKEN)
when served over HTTP; stdio is trusted local transport.
"""

from __future__ import annotations

import hmac
import os
from typing import Any


def create_server(name: str, instructions: str) -> Any:
    try:
        from mcp.server.mcpserver import MCPServer  # noqa: PLC0415 - SDK 2.x

        return MCPServer(name=name, instructions=instructions)
    except ImportError:
        from mcp.server.fastmcp import FastMCP  # noqa: PLC0415 - SDK 1.x

        return FastMCP(name=name, instructions=instructions)


def check_bearer(header_value: str | None, expected: str | None = None) -> bool:
    """Constant-time check of 'Bearer <token>' against the shared token."""
    expected = expected if expected is not None else os.environ.get("MCP_SHARED_BEARER_TOKEN", "")
    if not expected or not header_value or not header_value.startswith("Bearer "):
        return False
    return hmac.compare_digest(header_value.removeprefix("Bearer "), expected)


def run(server: Any) -> None:  # pragma: no cover - process entry point
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    server.run(transport=transport)
