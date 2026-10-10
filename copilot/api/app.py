"""FastAPI app: a thin shell over the run engine and services.

Run:  uvicorn copilot.api.app:create_app --factory --reload
Requires: pip install -e '.[api]'
"""

from __future__ import annotations

import logging
import os
from typing import Any

from copilot.adapters.jobsources import PastedJobParser
from copilot.api.auth import AuthError, verify_token
from copilot.api.runs import LocalRunEngine, RunEngine, RunNotFoundError, RunStateError
from copilot.api.schemas import (
    PastedJobRequest,
    ReviewRequest,
    RunResponse,
    StartRunRequest,
    StatusPatch,
)
from copilot.config.settings import Settings, load_settings
from copilot.container import Container, build_container
from copilot.graph.state import initial_state
from copilot.ports.repositories import InvalidTransitionError
from copilot.services.applications import ApplicationNotFoundError

log = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    container: Container | None = None,
    engine: RunEngine | None = None,
) -> Any:
    from fastapi import Depends, FastAPI, Header, HTTPException  # noqa: PLC0415
    from fastapi.middleware.cors import CORSMiddleware  # noqa: PLC0415

    settings = settings or load_settings(os.environ.get("APP_ROOT", "."))
    container = container or build_container(settings)
    engine = engine or LocalRunEngine(container.services)
    applications = container.services.applications
    jwt_secret = settings.secrets.jwt_secret.get_secret_value() if settings.secrets.jwt_secret else ""
    pasted = PastedJobParser()

    app = FastAPI(title="Job Application Copilot", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    def current_user(authorization: str = Header(default="")) -> str:
        """Bearer JWT -> user id. In dev without a secret, 'dev-user' is accepted."""
        if not authorization and settings.env == "dev":
            return "dev-user"
        if not authorization.startswith("Bearer "):
            raise HTTPException(401, "missing bearer token")
        try:
            return str(verify_token(authorization.removeprefix("Bearer "), jwt_secret)["sub"])
        except AuthError as exc:
            raise HTTPException(401, str(exc)) from exc

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "llm_profile": settings.llm.provider_profile,
            "embedding_profile": settings.llm.embedding_profile,
            "vector_store": settings.vector_store,
        }

    @app.post("/runs", response_model=RunResponse)
    def start_run(body: StartRunRequest, user: str = Depends(current_user)) -> RunResponse:
        try:
            state = initial_state(
                user, body.preferences, resume_text=body.resume_text, profile=body.profile
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return RunResponse.from_snapshot(engine.start(state))

    @app.get("/runs/{thread_id}", response_model=RunResponse)
    def get_run(thread_id: str, user: str = Depends(current_user)) -> RunResponse:
        try:
            snap = engine.get(thread_id)
        except RunNotFoundError as exc:
            raise HTTPException(404, "run not found") from exc
        if snap.state.get("user_id") != user:
            raise HTTPException(404, "run not found")
        return RunResponse.from_snapshot(snap)

    @app.post("/runs/{thread_id}/review", response_model=RunResponse)
    def review(thread_id: str, body: ReviewRequest, user: str = Depends(current_user)) -> RunResponse:
        get_run(thread_id, user)  # ownership check
        try:
            return RunResponse.from_snapshot(engine.resume(thread_id, body.decisions))
        except RunStateError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/jobs/parse")
    def parse_pasted(body: PastedJobRequest, user: str = Depends(current_user)) -> dict[str, Any]:
        posting = pasted.parse(body.text, url=body.url)
        container.services.indexing.index_job(posting)
        return posting.model_dump(mode="json")

    @app.get("/applications")
    def list_applications(user: str = Depends(current_user)) -> list[dict[str, Any]]:
        return [
            {"id": r.id, "job_id": r.job_id, "status": r.status, "artifact_uri": r.artifact_uri}
            for r in applications.list(user)
        ]

    @app.patch("/applications/{application_id}")
    def patch_application(
        application_id: str, body: StatusPatch, user: str = Depends(current_user)
    ) -> dict[str, Any]:
        try:
            record = applications.set_status(application_id, body.status, user_id=user)
        except ApplicationNotFoundError as exc:
            raise HTTPException(404, "application not found") from exc
        except InvalidTransitionError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"id": record.id, "status": record.status}

    return app


def main() -> None:  # pragma: no cover - entry point
    import uvicorn  # noqa: PLC0415

    uvicorn.run("copilot.api.app:create_app", factory=True, host="0.0.0.0", port=8000)  # noqa: S104
