---
name: run-job-copilot
description: Run, start, launch, smoke-test, drive or screenshot the Job Application Copilot (FastAPI API + Next.js web UI + LangGraph pipeline) locally, fully offline with no API keys or Docker. Use to verify a change end to end through the API or the browser UI.
---

# Run job-copilot

A FastAPI backend (`copilot/api`) runs the LangGraph pipeline, and a Next.js page (`web/`) drives it. The agent path is **offline**: `serve_offline.py` serves the API with the canned fake LLM (`copilot/adapters/llm/offline.py`), fixture jobs (no network) and in-memory vectors. `smoke.py` drives one run over HTTP, and `web_driver.py` drives the browser UI with Playwright and saves screenshots.

All paths are relative to the repo root (`D:\workspace\job-copilot`). Verified on Windows 11 with Git Bash, Python 3.13 and Node 24. Docker is **not** needed.

## Prerequisites (once)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -q -e ".[all,dev]"
.venv/Scripts/python -m pip install -q playwright      # web driver only; uses system Edge, no browser download
cp .env.example .env                                   # all dummy values; fine offline
(cd web && npm install --no-audit --no-fund)
```

## Run (agent path)

1. API, offline with canned LLM (background):
   ```bash
   .venv/Scripts/python .claude/skills/run-job-copilot/serve_offline.py 8000
   ```
2. Drive one full run over HTTP (health → POST /runs → approve → /applications). Prints `OK` and exits 0:
   ```bash
   .venv/Scripts/python .claude/skills/run-job-copilot/smoke.py
   ```
   Expect `score: 81 fixture:acme-101`, status `awaiting_review` and then `completed`, and a DOCX in `outputs/offline/dev-user/`.
3. Web UI (background). The API base defaults to `http://localhost:8000`, so no `.env.local` is needed:
   ```bash
   cd web && npm run dev -- -p 3000
   ```
4. Drive the UI (fill resume → Find and tailor → Approve all). Screenshots go to `outputs/screens/1-filled.png`, `2-review.png` and `3-approved.png`. **Restart the API first if `smoke.py` already ran** (see Gotchas):
   ```bash
   .venv/Scripts/python .claude/skills/run-job-copilot/web_driver.py
   ```
   Open `2-review.png` to check it: it should show "Matches 81 fixture:acme-101" and a "Review drafts" section.

The pipeline alone, with no server:
```bash
.venv/Scripts/python -m scripts.demo_run
```

## Test

```bash
.venv/Scripts/python -m pytest -p no:warnings      # 142 passed
```

## Run (human path / real LLM)

```bash
LLM_PROVIDER_PROFILE=fake EMBEDDING_PROFILE=hashing VECTOR_STORE=memory \
  .venv/Scripts/python -m uvicorn copilot.api.app:create_app --factory --port 8000
```
`smoke.py` passes against this too, but job search goes to the **live** Greenhouse/Lever boards in `config/companies.yaml` (network), so the scores and job ids vary. For real resume parsing, put `ANTHROPIC_API_KEY` in `.env` and drop `LLM_PROVIDER_PROFILE=fake`. (Not verified here: no key was available.)

## Gotchas

- **The fake LLM always parses to the same candidate.** `offline_llm()` returns a fixed profile (Asha Rao) whatever resume text you send, and fixed tailored bullets. That's fine for checking flow and UI, not for checking parsing.
- **One run per job per server lifetime.** The application tracker is in-memory per user (`dev-user` in dev with no auth header). After a job is approved, the next run on the same job fails with `InvalidTransitionError: cannot move approved -> tailored`. Restart `serve_offline.py` between `smoke.py` and `web_driver.py`, or between any two runs.
- A failed run shows as `Run failed: <error>` under "Matches" in the UI (`role="alert"`).
- `web/.env.local.example` doesn't exist, despite what the README says. That's harmless because `web/lib/api.ts` defaults to `http://localhost:8000`.
- The single `console: Failed to load resource: 404` from the web driver is the missing favicon. Ignore it.
- `npm install` warns `allow-scripts unrs-resolver postinstall`. The dev server works without it.
- `pytest` emits a LangGraph `UserWarning` about `config` typing in `copilot/graph/builder.py`. It's harmless.
- In Git Bash, foreground `sleep` may be blocked. To wait for the API: `for i in $(seq 1 20); do curl -sf localhost:8000/health >/dev/null && break; ping -n 2 127.0.0.1 >/dev/null; done`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `InvalidTransitionError: cannot move approved -> tailored` | Restart the API (in-memory tracker). |
| `web_driver.py`: `Review drafts` heading not found | Look at the page's `Run failed:` message, usually the error above. Restart the API and retry. |
| Port 8000 already in use | Stop the previous API (it's a background task). |
