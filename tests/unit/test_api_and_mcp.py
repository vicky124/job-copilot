import importlib.util
import tempfile
import time
import unittest
from datetime import UTC, datetime
from pathlib import Path

from copilot.adapters.calendar import IcsCalendarClient
from copilot.adapters.documents import MarkdownResumeRenderer
from copilot.adapters.tracing import InMemoryTracer
from copilot.api.auth import AuthError, sign_token, verify_token
from copilot.api.runs import LocalRunEngine, RunNotFoundError, RunStateError
from copilot.container import build_container
from copilot.domain.models import Decision, ReviewDecision
from copilot.graph import initial_state
from copilot.services.job_search import JobSearchService
from copilot.services.query_builder import SearchPreferences
from mcp_servers.calendar.server import create_event_tool
from mcp_servers.common import check_bearer
from mcp_servers.docgen.server import render_resume_tool
from mcp_servers.jobboard.tools import JobBoardTools
from tests.fakes import GOOD_TAILOR, StaticJobSource, make_job, make_profile, offline_settings, pipeline_llm

SECRET = "test-secret"


class AuthTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        token = sign_token({"sub": "u1", "aud": "job-copilot", "exp": time.time() + 60}, SECRET)
        self.assertEqual(verify_token(token, SECRET, audience="job-copilot")["sub"], "u1")

    def test_rejections(self) -> None:
        good = sign_token({"sub": "u1"}, SECRET)
        cases = {
            "bad signature": (good, "other-secret", None),
            "expired": (sign_token({"sub": "u1", "exp": time.time() - 3600}, SECRET), SECRET, None),
            "audience": (sign_token({"sub": "u1", "aud": "x"}, SECRET), SECRET, "job-copilot"),
            "no subject": (sign_token({"name": "x"}, SECRET), SECRET, None),
            "malformed": ("not.a.jwt.at.all", SECRET, None),
        }
        for name, (token, secret, aud) in cases.items():
            with self.subTest(name), self.assertRaises(AuthError):
                verify_token(token, secret, audience=aud)

    def test_alg_none_rejected(self) -> None:
        import base64
        import json

        def enc(d: dict) -> str:
            return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()

        forged = f"{enc({'alg': 'none'})}.{enc({'sub': 'admin'})}."
        with self.assertRaises(AuthError):
            verify_token(forged, SECRET)


class RunEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        container = build_container(
            offline_settings(Path(tmp.name)),
            llm=pipeline_llm(),
            job_sources=[StaticJobSource([make_job()])],
            tracer=InMemoryTracer(),
        )
        self.engine = LocalRunEngine(container.services)
        self.state = initial_state("u1", SearchPreferences(roles=["backend engineer"]), profile=make_profile())

    def test_start_pauses_then_resume_completes(self) -> None:
        snap = self.engine.start(self.state)
        self.assertEqual(snap.status, "awaiting_review")
        done = self.engine.resume(
            snap.thread_id, [ReviewDecision(job_id="test:j1", decision=Decision.APPROVE)]
        )
        self.assertEqual(done.status, "completed")
        self.assertIn("test:j1", done.state["artifacts"])

    def test_resume_requires_all_decisions_and_pending_run(self) -> None:
        snap = self.engine.start(self.state)
        with self.assertRaises(RunStateError):
            self.engine.resume(snap.thread_id, [ReviewDecision(job_id="other", decision=Decision.APPROVE)])
        self.engine.resume(snap.thread_id, [ReviewDecision(job_id="test:j1", decision=Decision.REJECT)])
        with self.assertRaises(RunStateError):
            self.engine.resume(snap.thread_id, [ReviewDecision(job_id="test:j1", decision=Decision.APPROVE)])

    def test_unknown_run(self) -> None:
        with self.assertRaises(RunNotFoundError):
            self.engine.get("nope")

    def test_failure_is_captured(self) -> None:
        bad = initial_state("u1", SearchPreferences(), profile=make_profile(experience=[]))
        snap = self.engine.start(bad)
        self.assertEqual(snap.status, "failed")
        self.assertIn("no target roles", snap.error)


@unittest.skipUnless(importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx"),
                     "fastapi/httpx not installed (pip install -e '.[api,http]')")
class ApiTest(unittest.TestCase):  # pragma: no cover - depends on optional deps
    def setUp(self) -> None:
        from fastapi.testclient import TestClient

        from copilot.api.app import create_app

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        settings = offline_settings(Path(tmp.name))
        container = build_container(
            settings, llm=pipeline_llm(), job_sources=[StaticJobSource([make_job()])], tracer=InMemoryTracer()
        )
        self.client = TestClient(create_app(settings, container))

    def test_run_review_flow(self) -> None:
        body = {"profile": make_profile().model_dump(mode="json"), "preferences": {"roles": ["backend engineer"]}}
        run = self.client.post("/runs", json=body).json()
        self.assertEqual(run["status"], "awaiting_review")
        done = self.client.post(
            f"/runs/{run['thread_id']}/review",
            json={"decisions": [{"job_id": "test:j1", "decision": "approve"}]},
        ).json()
        self.assertEqual(done["status"], "completed")
        apps = self.client.get("/applications").json()
        self.assertEqual(apps[0]["status"], "approved")


class McpToolsTest(unittest.TestCase):
    def test_jobboard_search_and_parse(self) -> None:
        tools = JobBoardTools(JobSearchService([StaticJobSource([make_job()])], InMemoryTracer()), ["static"])
        out = tools.search_jobs(["backend engineer"], ["Bengaluru"], ["hybrid"], 4, 10)
        self.assertEqual(len(out["jobs"]), 1)
        self.assertEqual(tools.list_sources(), ["static"])
        self.assertEqual(tools.parse_job("Title: SRE\nCompany: X\nKubernetes, Linux")["title"], "SRE")

    def test_docgen_renders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = render_resume_tool(
                MarkdownResumeRenderer(),
                Path(tmp),
                make_profile().model_dump(mode="json"),
                {"job_id": "test:j1", **GOOD_TAILOR},
            )
            self.assertTrue(Path(out["path"]).exists())

    def test_calendar_requires_timezone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = IcsCalendarClient(Path(tmp))
            out = create_event_tool(client, "Interview", "2026-10-14T10:00:00+05:30", "2026-10-14T11:00:00+05:30")
            self.assertTrue(out["event_id"].endswith("@job-copilot"))
            with self.assertRaises(ValueError):
                create_event_tool(client, "x", "2026-10-14T10:00:00", "2026-10-14T11:00:00")

    def test_bearer_check(self) -> None:
        self.assertTrue(check_bearer("Bearer abc", "abc"))
        self.assertFalse(check_bearer("Bearer abd", "abc"))
        self.assertFalse(check_bearer(None, "abc"))
        self.assertFalse(check_bearer("Bearer abc", ""))


class EvalScorersTest(unittest.TestCase):
    def test_scorers(self) -> None:
        from copilot.domain.models import TailoredResume
        from evals.scorers import ats_readability, faithfulness, keyword_coverage

        good = TailoredResume(job_id="j", **GOOD_TAILOR)
        self.assertEqual(faithfulness(make_profile(), good), 1.0)
        bad = good.model_copy(update={"bullets": ["Led Kafka migration"]})
        self.assertLess(faithfulness(make_profile(), bad), 1.0)
        self.assertEqual(keyword_coverage(["fastapi", "kafka"], good), 0.5)
        self.assertEqual(ats_readability(good), 1.0)

    def test_eval_cli_passes_gate(self) -> None:
        from evals.run_eval import main

        self.assertEqual(main([]), 0)


if __name__ == "__main__":
    unittest.main()
