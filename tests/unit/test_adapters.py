import io
import tempfile
import unittest
import urllib.error
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import mock

from copilot.adapters.calendar import IcsCalendarClient, to_ics
from copilot.adapters.documents import MarkdownResumeRenderer
from copilot.adapters.embeddings import HashingEmbedder, OllamaEmbedder, build_embedder
from copilot.adapters.embeddings.ollama import EmbeddingError
from copilot.adapters.http import UrllibHttpClient
from copilot.adapters.jobsources import (
    AdzunaJobSource,
    GreenhouseJobSource,
    LeverJobSource,
    PastedJobParser,
    strip_html,
)
from copilot.adapters.vectorstores import InMemoryVectorStore
from copilot.adapters.vectorstores.memory import CollectionNotFoundError, DimensionMismatchError
from copilot.config.settings import CompanyEntry, EmbeddingProfile
from copilot.domain.models import Chunk, TailoredResume, WorkMode
from copilot.ports.calendar import CalendarEvent
from copilot.ports.http import HttpError
from tests.fakes import GOOD_TAILOR, FakeHttpClient, make_profile, make_query, offline_settings

GH = CompanyEntry(name="Rubrik", ats="greenhouse", slug="rubrik", segment="gcc")
LV = CompanyEntry(name="CRED", ats="lever", slug="cred", segment="startup")
GH_URL = "https://boards-api.greenhouse.io/v1/boards/rubrik/jobs"
LV_URL = "https://api.lever.co/v0/postings/cred"

GH_PAYLOAD = {
    "jobs": [
        {
            "id": 7,
            "title": "Senior Backend Engineer",
            "location": {"name": "Bangalore, India"},
            "absolute_url": "https://rubrik.com/j/7",
            "first_published": "2026-09-01T10:00:00-04:00",
            "content": "&lt;p&gt;Requirements:&lt;/p&gt;&lt;ul&gt;&lt;li&gt;5+ years Python, Kubernetes&lt;/li&gt;&lt;/ul&gt;",
        },
        {"id": 8, "title": "Backend Engineer", "location": {"name": "Palo Alto, CA"}, "content": ""},
        {"id": 9, "title": "Account Executive", "location": {"name": "Mumbai"}, "content": ""},
    ]
}

LV_PAYLOAD = [
    {
        "id": "abc",
        "text": "backend engineer",
        "country": "IN",
        "workplaceType": "hybrid",
        "categories": {"location": "bengaluru", "allLocations": ["bengaluru"]},
        "descriptionPlain": "Build payment APIs. 3-5 years. CTC 20-30 LPA.",
        "lists": [{"text": "Requirements", "content": "<li>Go or Python</li><li>PostgreSQL</li>"}],
        "hostedUrl": "https://jobs.lever.co/cred/abc",
        "createdAt": 1790000000000,
    },
    {"id": "x", "text": "backend engineer", "country": "US", "categories": {"location": "NYC"}},
]


class StripHtmlTest(unittest.TestCase):
    def test_double_escaped_greenhouse_html(self) -> None:
        self.assertEqual(strip_html("&amp;lt;p&amp;gt;Hi &amp;amp; bye&amp;lt;/p&amp;gt;"), "Hi & bye")

    def test_block_tags_become_lines(self) -> None:
        self.assertEqual(strip_html("<ul><li>a</li><li>b</li></ul>"), "a\nb")


class GreenhouseTest(unittest.TestCase):
    def test_maps_india_jobs_matching_query(self) -> None:
        http = FakeHttpClient({GH_URL: GH_PAYLOAD})
        jobs = GreenhouseJobSource(http, [GH], []).search(make_query(locations=[]))
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertEqual((job.job_id, job.company, job.locations), ("greenhouse:rubrik-7", "Rubrik", ["Bengaluru"]))
        self.assertEqual(job.exp_min, 5.0)
        self.assertIn("kubernetes", job.required_skills)
        self.assertEqual(job.company_type.value, "gcc")

    def test_board_failure_is_skipped(self) -> None:
        http = FakeHttpClient({GH_URL: HttpError(GH_URL, 404)})
        self.assertEqual(GreenhouseJobSource(http, [GH], []).search(make_query()), [])


class LeverTest(unittest.TestCase):
    def test_maps_country_in_and_lists(self) -> None:
        jobs = LeverJobSource(FakeHttpClient({LV_URL: LV_PAYLOAD}), [LV], []).search(make_query())
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertIs(job.work_mode, WorkMode.HYBRID)
        self.assertEqual((job.ctc_min_lpa, job.ctc_max_lpa), (20.0, 30.0))
        self.assertIn("postgresql", job.required_skills)
        self.assertEqual(job.posted_at.tzinfo, UTC)


class AdzunaTest(unittest.TestCase):
    def test_params_and_inr_to_lpa(self) -> None:
        payload = {
            "results": [
                {
                    "id": 42,
                    "title": "Backend Engineer",
                    "company": {"display_name": "Acme"},
                    "location": {"display_name": "Bengaluru, Karnataka"},
                    "description": "Python services",
                    "redirect_url": "https://adzuna.in/42",
                    "salary_min": 1800000,
                    "salary_max": 2400000,
                    "created": "2026-10-01T08:00:00Z",
                }
            ]
        }
        http = FakeHttpClient({"https://api.adzuna.com/v1/api/jobs/in/search/1": payload})
        jobs = AdzunaJobSource(http, "id", "key").search(make_query(ctc_min_lpa=15))
        self.assertEqual((jobs[0].ctc_min_lpa, jobs[0].ctc_max_lpa), (18.0, 24.0))
        params = http.requests[0][1]
        self.assertEqual((params["app_id"], params["where"], params["salary_min"]), ("id", "Bengaluru", 1500000))

    def test_failure_returns_empty(self) -> None:
        http = FakeHttpClient({"https://api.adzuna.com": HttpError("u", 500)})
        self.assertEqual(AdzunaJobSource(http, "id", "key").search(make_query()), [])


class PastedJobTest(unittest.TestCase):
    def test_headers_portal_and_enrichment(self) -> None:
        text = "Title: Python Developer\nCompany: Zeta\nLocation: Gurgaon\nImmediate joiners. 4-6 yrs. Django."
        job = PastedJobParser().parse(text, url="https://www.naukri.com/job/123")
        self.assertEqual((job.source, job.title, job.company), ("naukri", "Python Developer", "Zeta"))
        self.assertEqual((job.locations, job.notice_period_days, job.exp_min), (["Gurugram"], 0, 4.0))

    def test_stable_id(self) -> None:
        parser = PastedJobParser()
        self.assertEqual(parser.parse("Some JD text").job_id, parser.parse("Some JD text").job_id)

    def test_empty_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PastedJobParser().parse("   ")


class UrllibHttpClientTest(unittest.TestCase):
    def test_retries_then_succeeds(self) -> None:
        sleeps: list[float] = []
        ok = mock.MagicMock()
        ok.__enter__.return_value.read.return_value = b'{"a": 1}'
        err = urllib.error.HTTPError("u", 503, "busy", {}, io.BytesIO())
        with mock.patch("urllib.request.urlopen", side_effect=[err, ok]):
            out = UrllibHttpClient(retries=2, sleep=sleeps.append).get_json("https://x/y", params={"q": "a b"})
        self.assertEqual(out, {"a": 1})
        self.assertEqual(sleeps, [0.5])

    def test_non_retryable_raises_immediately(self) -> None:
        err = urllib.error.HTTPError("u", 404, "missing", {}, io.BytesIO())
        with mock.patch("urllib.request.urlopen", side_effect=err) as urlopen, self.assertRaises(HttpError):
            UrllibHttpClient(sleep=lambda _: None).get_json("https://x")
        self.assertEqual(urlopen.call_count, 1)


class EmbeddingsTest(unittest.TestCase):
    def test_hashing_deterministic_and_normalised(self) -> None:
        emb = HashingEmbedder(64)
        a, b = emb.embed_query("python redis"), emb.embed_query("python redis")
        self.assertEqual(a, b)
        self.assertAlmostEqual(sum(x * x for x in a), 1.0, places=6)

    def test_ollama_uses_prefixes_and_checks_dims(self) -> None:
        sent = {}

        def post(url, payload):
            sent.update(payload)
            return {"embeddings": [[0.1, 0.2, 0.3]]}

        profile = EmbeddingProfile(
            model="nomic-embed-text", dimensions=3, collection_suffix="s",
            doc_prefix="search_document: ", query_prefix="search_query: ",
        )
        OllamaEmbedder(profile, post).embed_query("hi")
        self.assertEqual(sent["input"], ["search_query: hi"])
        bad = EmbeddingProfile(model="m", dimensions=768, collection_suffix="s")
        with self.assertRaises(EmbeddingError):
            OllamaEmbedder(bad, post).embed_documents(["x"])

    def test_factory(self) -> None:
        self.assertIsInstance(build_embedder(offline_settings().llm), HashingEmbedder)


class InMemoryVectorStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryVectorStore()
        self.store.ensure_collection("c", 2)
        self.store.upsert(
            "c",
            [Chunk(id="a", text="a", metadata={"g": "1"}), Chunk(id="b", text="b", metadata={"g": "2"})],
            [[1.0, 0.0], [0.0, 1.0]],
        )

    def test_search_orders_and_filters(self) -> None:
        self.assertEqual([h.chunk.id for h in self.store.search("c", [1.0, 0.1], limit=2)], ["a", "b"])
        self.assertEqual([h.chunk.id for h in self.store.search("c", [1.0, 0.0], limit=5, filters={"g": "2"})], ["b"])

    def test_delete_group(self) -> None:
        self.store.delete_group("c", "g", "1")
        self.assertEqual(self.store.count("c"), 1)

    def test_dimension_and_collection_guards(self) -> None:
        with self.assertRaises(DimensionMismatchError):
            self.store.ensure_collection("c", 3)
        with self.assertRaises(DimensionMismatchError):
            self.store.upsert("c", [Chunk(id="z", text="z")], [[1.0, 2.0, 3.0]])
        with self.assertRaises(CollectionNotFoundError):
            self.store.search("missing", [1.0], limit=1)


class DocumentsAndCalendarTest(unittest.TestCase):
    def test_markdown_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            doc = MarkdownResumeRenderer().render_resume(
                make_profile(), TailoredResume(job_id="greenhouse:x/1", **GOOD_TAILOR), Path(tmp)
            )
            text = doc.path.read_text()
        self.assertEqual(doc.path.name, "Asha-Rao_greenhouse-x-1.md")
        self.assertIn("# Asha Rao", text)
        self.assertIn("- Built FastAPI services", text)

    def test_ics_content_and_idempotency(self) -> None:
        start = datetime(2026, 10, 14, 10, 0, tzinfo=UTC)
        event = CalendarEvent("Interview, Acme", start, start + timedelta(hours=1), idempotency_key="k1")
        ics = to_ics(event, now=start)
        self.assertIn("SUMMARY:Interview\\, Acme", ics)
        self.assertIn("DTSTART:20261014T100000Z", ics)
        with tempfile.TemporaryDirectory() as tmp:
            client = IcsCalendarClient(Path(tmp))
            self.assertEqual(client.create_event(event), client.create_event(event))
            self.assertEqual(len(list(Path(tmp).glob("*.ics"))), 1)

    def test_event_end_after_start(self) -> None:
        start = datetime(2026, 10, 14, tzinfo=UTC)
        with self.assertRaises(ValueError):
            CalendarEvent("x", start, start)


if __name__ == "__main__":
    unittest.main()
