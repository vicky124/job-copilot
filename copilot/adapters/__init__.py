"""Adapters implement ports with concrete libraries.

Heavy libraries (litellm, qdrant-client, langfuse, httpx) are imported inside the
adapter that needs them, so importing ``copilot`` never requires them.
"""
