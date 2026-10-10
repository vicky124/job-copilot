# Job Application Copilot

An AI assistant for job hunting in India. It parses your resume, finds matching jobs, scores how well you fit each one, tailors your resume per job **without inventing anything**, waits for your approval, then generates the document and prepares interview questions.

It is built to exercise the full modern LLM-app stack: **LLM orchestration with LangGraph**, **RAG over Qdrant**, **MCP servers** for every external capability, **provider-agnostic LLM integration**, and **observability with Langfuse** including A/B prompt evaluation.

> Status: v0.1, a working skeleton. The whole pipeline runs end to end offline today (fake LLM, in-memory vectors, fixture jobs). Real providers plug in through config.

---

## Contents

1. [What it does](#what-it-does)
2. [Quick start (5 minutes, no keys)](#quick-start-5-minutes-no-keys)
3. [Full local setup](#full-local-setup)
4. [Configuration](#configuration)
5. [Architecture](#architecture)
6. [The LangGraph pipeline](#the-langgraph-pipeline)
7. [RAG design](#rag-design)
8. [MCP servers](#mcp-servers)
9. [Observability and A/B evals](#observability-and-ab-evals)
10. [API](#api)
11. [Project layout](#project-layout)
12. [Coding guidelines and SOLID](#coding-guidelines-and-solid)
13. [Testing](#testing)
14. [Security and privacy](#security-and-privacy)
15. [Roadmap](#roadmap)

---

## What it does

| Step | What happens | Tech |
| --- | --- | --- |
| Parse resume | PDF/DOCX/text → structured `CandidateProfile` | LLM structured output (small tier) |
| Search jobs | Adzuna India API, Greenhouse and Lever company boards, pasted Naukri/LinkedIn JDs | `jobboard-mcp`, fan-out with failure isolation |
| Normalise | Cities (Bangalore → Bengaluru), CTC in LPA, notice period, experience band, work mode, skills | Pure domain functions |
| Index | Job descriptions, your achievements, interview questions | Qdrant, local `nomic-embed-text` |
| Score fit | 0–100 from skills, evidence, experience, notice period, CTC; LLM writes the rationale | RAG requirement ↔ evidence pairs |
| Tailor | Rewrites summary and bullets for the job | LLM (strong tier), A/B prompt variants |
| Verify | Blocks any new skill, number, employer or credential not in your resume | Deterministic guardrail |
| Review | **You** approve, edit or reject each draft. Nothing is generated before that. | LangGraph `interrupt()` |
| Generate | ATS-friendly DOCX resume | `docgen-mcp` |
| Prep | Likely questions by round + STAR answer drafts from your real experience | RAG over a question bank |
| Schedule | Interview and prep-time events | `calendar-mcp` (.ics now, Google Calendar next) |

---

## Quick start (5 minutes, no keys)

Requires **Python 3.12+**. Works on Windows, macOS and Linux.

```bash
git clone <your-repo-url> job-copilot   # or open the folder you already have
cd job-copilot

python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# macOS / Linux:         source .venv/bin/activate

pip install -e ".[dev,documents]"

python -m scripts.demo_run        # whole pipeline, offline
python -m unittest discover -s tests -t .   # or: pytest
```

The demo runs with a fake LLM, hashing embeddings, an in-memory vector store and three sample jobs. Expected output:

```
path      : parse_resume -> build_query -> search_jobs -> index_jobs -> score_fit -> select_jobs -> tailor_resume -> verify_claims -> human_review -> generate_docs -> interview_prep
score     :  81  fixture:acme-101  gaps=['django', 'kafka']
resume    : fixture:acme-101 -> outputs/demo/demo-user/Asha-Rao_fixture-acme-101.docx
prep      : fixture:acme-101: 9 questions
```

---

## Full local setup

### 1. Credentials

```bash
cp .env.example .env              # Windows: copy .env.example .env
```

**Every value in `.env.example` is a dummy placeholder.** Replace only what you use:

| Need | Keys | Where to get them |
| --- | --- | --- |
| Default LLM (Anthropic) | `ANTHROPIC_API_KEY` | console.anthropic.com |
| Other LLMs (optional) | `OPENAI_API_KEY`, `GEMINI_API_KEY`, `AZURE_API_*` | provider consoles |
| Adzuna India jobs | `ADZUNA_APP_ID`, `ADZUNA_APP_KEY` | developer.adzuna.com (free tier) |
| Langfuse tracing | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | your self-hosted Langfuse → Settings → API keys |
| API ↔ web auth | `APP_JWT_SECRET` | `openssl rand -hex 32` (or `python -c "import secrets;print(secrets.token_hex(32))"`) |

Keys still containing `dummy` are treated as unset: Adzuna is skipped, Langfuse is off. In `APP_ENV=prod` the app **refuses to start** with dummy secrets.

### 2. Infrastructure (Docker)

```bash
docker compose up -d qdrant postgres redis minio ollama
docker compose exec ollama ollama pull nomic-embed-text      # local embedding model (768-dim)
docker compose --profile observability up -d                 # self-hosted Langfuse on :3001
```

### 3. Install everything and run

```bash
pip install -e ".[all,dev]"
uvicorn copilot.api.app:create_app --factory --reload        # API on :8000, docs at /docs
```

MCP servers (stdio for Claude Desktop / local agents, or HTTP):

```bash
python -m mcp_servers.jobboard.server
MCP_TRANSPORT=streamable-http python -m mcp_servers.jobboard.server
```

Web app:

```bash
cd web
cp .env.local.example .env.local
npm install
npm run dev                                                  # http://localhost:3000
```

### 4. Validate the company boards

```bash
python -m scripts.validate_companies --only-unseen
```

Checks every slug in `config/companies.yaml` and counts current India postings.

---

## Configuration

Non-secret settings live in YAML (committed). Secrets live in `.env` (never committed). Environment variables override YAML.

### `config/llm.yaml` — models per tier

Nodes ask for a **tier**, never a vendor:

| Tier | Used by | Default (Anthropic profile) |
| --- | --- | --- |
| `small` | parse resume, JD enrichment | `anthropic/claude-haiku-5-5` |
| `medium` | fit rationale, interview prep | `anthropic/claude-sonnet-5-5` |
| `strong` | resume tailoring | `anthropic/claude-opus-5-5` |

Switch provider without touching code:

```bash
LLM_PROVIDER_PROFILE=openai     # anthropic | openai | gemini | azure | ollama | fake
```

Fallbacks (`strong → medium → small`) and a per-run cost ceiling (`budget.per_run_usd`) are in the same file. API keys from `.env` are passed to each call directly, so nothing needs exporting.

### Embeddings

```bash
EMBEDDING_PROFILE=local         # local (nomic-embed-text, 768) | openai (1536) | hashing (tests)
```

Each profile writes to its own Qdrant collections (`job_descriptions_v1_nomic768`, …). Switching means: run the re-index, then flip the setting. Vectors from different models are never mixed, and the app refuses to start on a dimension mismatch.

### `config/app.yaml` — pipeline tuning

`fit_threshold`, `max_jobs_to_tailor`, retry limits, retrieval `top_k`, and A/B experiment weights.

### `config/companies.yaml` — Greenhouse and Lever boards

20 India-hiring companies (16 Greenhouse, 4 Lever), slugs checked on 10 Oct 2026. Toggle an adapter or a company with `enabled: false`. Postings outside `location_filter` are dropped.

### Fully offline mode

```bash
LLM_PROVIDER_PROFILE=fake EMBEDDING_PROFILE=hashing VECTOR_STORE=memory
```

---

## Architecture

Ports and adapters (hexagonal). Dependencies point **inward only**.

```
            ┌────────────────────────────────────────────────────┐
 delivery   │  FastAPI (copilot/api)   MCP servers   Next.js web │
            └──────────────┬─────────────────────────────────────┘
                           │
 orchestration   LangGraph StateGraph (copilot/graph) — nodes call services
                           │
 use cases       services/: resume_parser, job_search, indexing, retrieval,
                 fit_scorer, tailor, claim_verifier, interview_prep, experiments
                           │ depend only on ↓
 interfaces      ports/: LLMClient, Embedder, VectorStore, JobSource, Tracer,
                 PromptStore, DocumentRenderer, CalendarClient, Repository
                           │ implemented by ↓
 infrastructure  adapters/: LiteLLM, Ollama, Qdrant, Greenhouse, Lever, Adzuna,
                 Langfuse, DOCX, ICS, in-memory fakes
                           │
 domain          domain/: models, India normalisation, skills, matching (pure)

 composition root  copilot/container.py — the only place that picks adapters
```

| Layer | Components |
| --- | --- |
| Presentation | Next.js (App Router, TypeScript) |
| API | FastAPI, JWT from Auth.js, run start / review / resume |
| Orchestration | LangGraph `StateGraph`, Postgres checkpointer, `interrupt()` |
| Intelligence | LiteLLM gateway, tiered models, Langfuse prompt registry |
| Knowledge | Qdrant, local `nomic-embed-text` via Ollama |
| Tools | `jobboard-mcp`, `docgen-mcp`, `calendar-mcp` |
| Data | Postgres, Redis, MinIO/S3 |
| Observability | Langfuse (self-hosted), OpenTelemetry, Prometheus/Grafana |

---

## The LangGraph pipeline

Declared once in `copilot/graph/spec.py`. Both the LangGraph build (`graph/builder.py`) and the dependency-free local runner (`graph/runner.py`) are generated from it, so production and tests use the same edges.

```
parse_resume → build_query → search_jobs → index_jobs → score_fit
score_fit ──(nothing above threshold, loops left)──→ build_query (widened)
score_fit → select_jobs → tailor_resume → verify_claims
verify_claims ──(violations, retries left)──→ tailor_resume
verify_claims → human_review            ← run pauses here
human_review ──edit──→ verify_claims → human_review   (your text is flagged, never rewritten)
human_review ──approve──→ generate_docs → interview_prep → END
human_review ──reject all──→ END
```

Guarantees, all covered by tests:

- Nothing is generated before an explicit approve.
- A draft that invents facts is sent back to the tailor (max 2 retries), then shown to you flagged.
- Text you edit is never rewritten by the model; problems in it are shown as flags.
- Approved or rejected drafts are never touched again.
- A failing job source never fails the run.

---

## RAG design

| Collection | Content | Chunk unit |
| --- | --- | --- |
| `job_descriptions_*` | JDs from all sources | Requirements: one bullet per chunk. Other sections: ~300 words, 40 overlap |
| `candidate_profile_*` | Your achievements | One bullet per chunk, tagged with role and company |
| `interview_questions_*` | Question bank | One question, tagged with round type, skill, difficulty |

**Fit scoring**: fetch a job's requirement chunks, then for each requirement search *your* achievements (filtered by `user_id`) for evidence. The model sees requirement ↔ evidence pairs, not whole documents. The score is computed deterministically from those pairs, so it is explainable and stable when prompts change.

---

## MCP servers

| Server | Tools | Side effects |
| --- | --- | --- |
| `jobboard-mcp` | `search_jobs`, `parse_job`, `list_sources` | None (read-only) |
| `docgen-mcp` | `render_resume` | Creates files — only called after approval |
| `calendar-mcp` | `create_event` (idempotency key) | Writes `.ics` now; Google Calendar via the same port next |

Tool logic lives in plain functions (unit-tested without MCP). `mcp_servers/common.py` supports both the MCP Python SDK 1.x (`FastMCP`) and 2.x (`MCPServer`).

Use with Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "jobboard": {
      "command": "python",
      "args": ["-m", "mcp_servers.jobboard.server"],
      "cwd": "C:/Users/User/workspace/job-copilot"
    }
  }
}
```

---

## Observability and A/B evals

- Every LLM call is wrapped by `TracedLLM`: one span with tier, model, tokens, cost, node, prompt version and variant.
- PII (email, Indian mobile numbers, PAN, Aadhaar) is redacted from every span attribute before export.
- `BudgetGuardLLM` stops a run once its spend passes `budget.per_run_usd`.
- Prompts are versioned YAML in `prompts/` with labels (`production`, `variant-b`) and seed Langfuse prompt management.
- `ABRouter` assigns each user a sticky variant from `config/app.yaml` weights.
- `python -m evals.run_eval` scores faithfulness, JD keyword coverage and ATS readability on `evals/datasets/golden.jsonl`. CI fails if faithfulness drops below **0.95**.

---

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Active LLM profile, embedding profile, vector store |
| POST | `/runs` | Start a run from resume text or a profile + preferences |
| GET | `/runs/{thread_id}` | Run status, scores, drafts awaiting review, artifacts |
| POST | `/runs/{thread_id}/review` | Approve / edit / reject drafts and resume the run |
| POST | `/jobs/parse` | Parse and index a pasted JD |
| GET | `/applications` | Your application tracker |
| PATCH | `/applications/{id}` | Move status (applied, interviewing, offer…) — invalid moves return 409 |

Auth: `Authorization: Bearer <HS256 JWT>` signed with `APP_JWT_SECRET`. In `dev` without a header you are `dev-user`.

---

## Project layout

```
job-copilot/
├── config/                 llm.yaml, app.yaml, companies.yaml
├── prompts/                versioned prompt YAML (A/B labels)
├── copilot/
│   ├── domain/             models, india.py, skills.py, matching.py   (pure)
│   ├── ports/              Protocol interfaces
│   ├── adapters/           llm, embeddings, vectorstores, jobsources, tracing, documents, calendar…
│   ├── services/           use cases
│   ├── graph/              spec, state, nodes, gates, runner, LangGraph builder
│   ├── api/                FastAPI app, run engines, JWT auth
│   ├── config/settings.py  typed settings + secrets
│   └── container.py        composition root
├── mcp_servers/            jobboard, docgen, calendar
├── evals/                  scorers, golden dataset, CI gate
├── scripts/                demo_run, validate_companies, sample_data/
├── tests/unit/             139 unit tests; tests/fakes/ shared doubles
├── web/                    Next.js app
├── docker-compose.yml, Dockerfile, Makefile, .github/workflows/ci.yml
└── .env.example            dummy credentials
```

---

## Coding guidelines and SOLID

**Standards**: Python 3.12, full type hints (`mypy --strict`), `ruff` (pycodestyle, pyflakes, isort, bugbear, pyupgrade), Pydantic v2 for every boundary, frozen domain models, no I/O in the domain, no global state, logging instead of print outside scripts, secrets only via `SecretStr`.

| Principle | Where you can see it |
| --- | --- |
| **S**ingle responsibility | `enrich.py` derives India fields; sources only fetch and map. `FitScorer` computes, the LLM only explains. Nodes only call services. |
| **O**pen/closed | New job board = new `JobSource` class, no change to search or graph. Tracing and budgets are decorators around any `LLMClient`. |
| **L**iskov substitution | `FakeLLMClient`/`LiteLLMClient`, `InMemoryVectorStore`/`QdrantVectorStore`, `HashingEmbedder`/`OllamaEmbedder` are interchangeable — the test suite runs the real pipeline on the fakes. |
| **I**nterface segregation | Small protocols: `HttpClient` has one method; `CalendarClient` one; `JobSource` two. |
| **D**ependency inversion | Services depend on `ports/`; only `container.py` names concrete adapters. |

Contribution rules: one concern per PR, tests for every behaviour change, no new dependency in `domain/` or `services/`, keep optional libraries imported inside their adapter.

---

## Testing

```bash
python -m unittest discover -s tests -t .     # no extra packages needed
pytest --cov=copilot                          # with dev extras
make lint typecheck test demo eval
```

- **139 unit tests**, all offline, under a second.
- Fakes in `tests/fakes/`: scripted LLM, static job source, fake HTTP client, builders for profiles and jobs.
- Covered: config validation and prod secret checks, India parsing (CTC, notice, cities, experience), skills, matching, every job source mapping, HTTP retry/backoff, embeddings, vector store guards, structured-output retry, prompt rendering for every shipped prompt, fit scoring, the no-fabrication guardrail, tailoring with PII redaction, A/B bucketing, application status transitions, JWT (including `alg: none` forgery), every graph router, and end-to-end runs: happy path, fabrication retry, flagged review, query widening, pause → edit → re-review → approve.
- Two tests skip unless `langgraph` / `fastapi` are installed; they compile the real LangGraph graph and exercise the HTTP API.

---

## Security and privacy

- Contact details are stripped before any LLM call; PII is redacted from traces.
- Job descriptions are untrusted input: they only reach the model as data and never decide tool calls; write tools run only after approval.
- No scraping of Naukri/LinkedIn; users paste JDs. Adzuna results carry "Jobs by Adzuna" attribution.
- Built for India's DPDP Act 2023: consent at upload, deletion on request, India-region hosting. Check the DPDP Rules in force before launch.
- Run status checks enforce ownership; other users' runs and applications return 404.

---

## Roadmap

- [x] Domain, ports, adapters, services, graph, API, MCP servers, evals, 139 tests
- [ ] Postgres repository + Alembic migrations (port already defined)
- [ ] `LangGraphRunEngine` wired to `PostgresSaver` in the API
- [ ] Langfuse prompt store adapter (same `PromptStore` port)
- [ ] Google Calendar adapter for `calendar-mcp`
- [ ] Next.js: matches screen with evidence drawer, side-by-side diff review, tracker kanban, prep screen
- [ ] Finalise `config/companies.yaml` with your own target companies
- [ ] Reranker (Cohere or `bge-reranker`) and hybrid BM25 search

Architecture document: the "Job Application Copilot — Architecture Document" in Claude Docs.
