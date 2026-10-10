"""Graph routing and end-to-end runs with fakes (no network, no keys)."""

import tempfile
import unittest
from pathlib import Path

from copilot.adapters.tracing import InMemoryTracer
from copilot.config.settings import PipelineConfig
from copilot.container import build_container
from copilot.domain.models import Decision, FitScore, ReviewDecision
from copilot.graph import AutoApproveGate, CallbackReviewGate, LocalPipelineRunner, initial_state
from copilot.graph.gates import PausingReviewGate
from copilot.graph.runner import RecursionLimitError
from copilot.graph.spec import END, route_after_review, route_after_score, route_after_verify
from copilot.graph.state import apply_update
from copilot.services.query_builder import SearchPreferences
from tests.fakes import (
    FABRICATED_TAILOR,
    GOOD_TAILOR,
    StaticJobSource,
    make_job,
    make_profile,
    offline_settings,
    pipeline_llm,
)

CFG = PipelineConfig(fit_threshold=60, max_query_widen_loops=2, max_tailor_retries=2)


class RouterTest(unittest.TestCase):
    def test_score_router_widens_then_gives_up(self) -> None:
        low = {"scored": [FitScore(job_id="j", score=10)], "query_loops": 0}
        self.assertEqual(route_after_score(low, CFG), "build_query")
        self.assertEqual(route_after_score({**low, "query_loops": 2}, CFG), "select_jobs")
        self.assertEqual(route_after_score({"scored": [FitScore(job_id="j", score=90)]}, CFG), "select_jobs")

    def test_verify_router_retries_until_limit(self) -> None:
        from copilot.domain.models import ClaimViolation

        v = {"j": [ClaimViolation(bullet="b", reason="r")]}
        self.assertEqual(route_after_verify({"violations": v, "tailor_attempts": {"j": 1}}, CFG), "tailor_resume")
        self.assertEqual(route_after_verify({"violations": v, "tailor_attempts": {"j": 3}}, CFG), "human_review")
        self.assertEqual(route_after_verify({"violations": {"j": []}}, CFG), "human_review")

    def test_review_router(self) -> None:
        def state(decision: Decision) -> dict:
            kwargs = {"edited_summary": "x"} if decision is Decision.EDIT else {}
            return {"pending_review": ["j"], "review": {"j": ReviewDecision(job_id="j", decision=decision, **kwargs)}}

        self.assertEqual(route_after_review(state(Decision.EDIT), CFG), "verify_claims")
        self.assertEqual(route_after_review(state(Decision.APPROVE), CFG), "generate_docs")
        self.assertEqual(route_after_review(state(Decision.REJECT), CFG), END)

    def test_apply_update_merges_and_appends(self) -> None:
        s = apply_update({"drafts": {"a": 1}, "errors": ["e1"]}, {"drafts": {"b": 2}, "errors": ["e2"], "x": 1})
        self.assertEqual(s, {"drafts": {"a": 1, "b": 2}, "errors": ["e1", "e2"], "x": 1})


class PipelineTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = offline_settings(Path(self.tmp.name))
        self.tracer = InMemoryTracer()

    def build(self, llm=None, jobs=None):
        jobs = jobs if jobs is not None else [make_job()]
        return build_container(
            self.settings,
            llm=llm or pipeline_llm(),
            job_sources=[StaticJobSource(jobs)],
            tracer=self.tracer,
        )

    def state(self):
        return initial_state("u1", SearchPreferences(roles=["backend engineer"]), profile=make_profile())

    def runner(self, container, gate):
        return LocalPipelineRunner(container.nodes(gate), self.settings.app.pipeline, self.tracer)


class EndToEndTest(PipelineTestCase):
    def test_happy_path_generates_resume_and_prep(self) -> None:
        container = self.build()
        result = self.runner(container, AutoApproveGate()).run(self.state())
        self.assertTrue(result.finished)
        self.assertEqual(result.path[-3:], ["human_review", "generate_docs", "interview_prep"])
        artifact = Path(result.state["artifacts"]["test:j1"])
        self.assertTrue(artifact.exists())
        self.assertIn("test:j1", result.state["interview_packs"])
        app = container.services.applications.list("u1")[0]
        self.assertEqual(app.status.value, "approved")
        self.assertTrue(self.tracer.named("node.score_fit"))

    def test_fabrication_is_retried_then_fixed(self) -> None:
        container = self.build(llm=pipeline_llm([FABRICATED_TAILOR, GOOD_TAILOR]))
        result = self.runner(container, AutoApproveGate()).run(self.state())
        self.assertEqual(result.path.count("tailor_resume"), 2)
        self.assertEqual(result.state["violations"]["test:j1"], [])
        self.assertEqual(result.state["drafts"]["test:j1"].bullets, GOOD_TAILOR["bullets"])

    def test_persistent_fabrication_goes_to_human_flagged(self) -> None:
        seen = {}

        def review(payload):
            seen.update(payload)
            return [{"job_id": "test:j1", "decision": "reject"}]

        container = self.build(llm=pipeline_llm([FABRICATED_TAILOR]))
        result = self.runner(container, CallbackReviewGate(review)).run(self.state())
        self.assertEqual(result.path.count("tailor_resume"), 3)  # 1 + max_tailor_retries
        self.assertTrue(seen["drafts"][0]["violations"])
        self.assertEqual(result.path[-1], "human_review")
        self.assertEqual(result.state.get("artifacts", {}), {})

    def test_no_match_widens_query_then_stops(self) -> None:
        container = self.build(jobs=[])
        result = self.runner(container, AutoApproveGate()).run(self.state())
        self.assertEqual(result.path.count("build_query"), 3)
        self.assertEqual(result.path[-1], "select_jobs")
        self.assertEqual(result.state["query_loops"], 2)

    def test_pause_then_resume_with_edit(self) -> None:
        container = self.build()
        gate = PausingReviewGate()
        runner = self.runner(container, gate)
        paused = runner.run(self.state())
        self.assertEqual(paused.paused_at, "human_review")
        self.assertNotIn("artifacts", paused.state)

        # The user edits; the edit adds a skill not in the profile -> flagged, not rewritten.
        gate.decisions = {
            "test:j1": ReviewDecision(job_id="test:j1", decision=Decision.EDIT, edited_bullets=["Shipped Kafka"])
        }
        again = runner.run(paused.state, start="human_review")
        self.assertEqual(again.paused_at, "human_review")
        self.assertEqual(again.state["drafts"]["test:j1"].bullets, ["Shipped Kafka"])
        self.assertTrue(again.review_payload["drafts"][0]["violations"])

        gate.decisions = {"test:j1": ReviewDecision(job_id="test:j1", decision=Decision.APPROVE)}
        done = runner.run(again.state, start="human_review")
        self.assertTrue(done.finished)
        self.assertIn("test:j1", done.state["artifacts"])
        self.assertEqual(done.path.count("tailor_resume"), 0)  # user text never re-tailored

    def test_recursion_limit(self) -> None:
        container = self.build()
        runner = LocalPipelineRunner(
            container.nodes(AutoApproveGate()), self.settings.app.pipeline, self.tracer, recursion_limit=3
        )
        with self.assertRaises(RecursionLimitError):
            runner.run(self.state())


class LangGraphBuildTest(PipelineTestCase):
    def test_compiles_when_langgraph_installed(self) -> None:
        try:
            import langgraph  # noqa: F401
        except ImportError:
            self.skipTest("langgraph not installed (pip install -e '.[graph]')")
        from copilot.graph.builder import build_graph

        graph = build_graph(self.build().nodes(AutoApproveGate()), self.settings.app.pipeline)
        out = graph.invoke(self.state())
        self.assertIn("test:j1", out["artifacts"])


if __name__ == "__main__":
    unittest.main()
