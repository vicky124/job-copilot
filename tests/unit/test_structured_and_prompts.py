import json
import tempfile
import unittest
from pathlib import Path

from pydantic import BaseModel

from copilot.adapters.llm import FakeLLMClient
from copilot.adapters.prompts import FilePromptStore
from copilot.ports.llm import LLMResponse
from copilot.ports.prompts import PromptNotFoundError
from copilot.services.structured import StructuredOutputError, complete_structured, extract_json
from tests.fakes import ROOT


class _Out(BaseModel):
    name: str
    n: int


class ExtractJsonTest(unittest.TestCase):
    def test_strips_fences_and_prose(self) -> None:
        self.assertEqual(extract_json('Sure!\n```json\n{"a": 1}\n```'), '{"a": 1}')

    def test_no_json(self) -> None:
        with self.assertRaises(StructuredOutputError):
            extract_json("no braces here")


class CompleteStructuredTest(unittest.TestCase):
    def test_parses_valid_json(self) -> None:
        llm = FakeLLMClient(default=json.dumps({"name": "a", "n": 1}))
        out = complete_structured(llm, "small", "sys", "user", _Out)
        self.assertEqual(out, _Out(name="a", n=1))
        self.assertTrue(llm.calls[0].json_mode)

    def test_retries_once_with_error(self) -> None:
        replies = iter(['{"name": "a"}', '{"name": "a", "n": 2}'])

        class Seq(FakeLLMClient):
            def complete(self, tier, messages, *, json_mode=False, tags=None):
                self.calls.append(messages)
                return LLMResponse(text=next(replies), model="seq")

        llm = Seq()
        self.assertEqual(complete_structured(llm, "small", "s", "u", _Out).n, 2)
        self.assertIn("invalid", llm.calls[1][-1].content)

    def test_gives_up_after_retries(self) -> None:
        with self.assertRaises(StructuredOutputError):
            complete_structured(FakeLLMClient(default="nope"), "small", "s", "u", _Out, retries=1)


class FilePromptStoreTest(unittest.TestCase):
    def test_labels_resolve_to_versions(self) -> None:
        store = FilePromptStore(ROOT / "prompts")
        self.assertEqual(store.get("tailor_resume").version, "1.0.0")
        self.assertEqual(store.get("tailor_resume", "variant-b").version, "1.1.0")

    def test_unknown_label_falls_back_to_production(self) -> None:
        store = FilePromptStore(ROOT / "prompts")
        self.assertEqual(store.get("parse_resume", "variant-z").label, "production")

    def test_missing_prompt(self) -> None:
        with self.assertRaises(PromptNotFoundError):
            FilePromptStore(ROOT / "prompts").get("nope")

    def test_render_fills_and_keeps_literal_braces(self) -> None:
        system, user = FilePromptStore(ROOT / "prompts").get("parse_resume").render(resume_text="CV")
        self.assertIn("CV", user)
        self.assertIn("{title, company", user)

    def test_render_missing_variable_fails_loudly(self) -> None:
        with self.assertRaises(KeyError):
            FilePromptStore(ROOT / "prompts").get("parse_resume").render()

    def test_every_shipped_prompt_renders(self) -> None:
        """Guards against an unescaped brace or a misspelt placeholder in prompts/."""
        variables = {
            "parse_resume": {"resume_text": "x"},
            "score_fit": dict.fromkeys(["title", "company", "matched", "gaps", "evidence_ratio"], "x"),
            "tailor_resume": dict.fromkeys(
                ["job_title", "company", "requirements", "evidence", "summary", "bullets", "skills", "feedback"],
                "x",
            ),
            "interview_prep": dict.fromkeys(["job_title", "company", "experience", "questions"], "x"),
        }
        store = FilePromptStore(ROOT / "prompts")
        for name, vars_ in variables.items():
            for label in ("production", "variant-b"):
                with self.subTest(prompt=name, label=label):
                    store.get(name, label).render(**vars_)

    def test_label_without_production(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "p.yaml").write_text(
                "versions:\n  '1':\n    labels: [beta]\n    system: s\n    user: u\n"
            )
            with self.assertRaises(PromptNotFoundError):
                FilePromptStore(Path(tmp)).get("p", "production")


if __name__ == "__main__":
    unittest.main()
