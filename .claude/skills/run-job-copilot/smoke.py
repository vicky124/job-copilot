"""Drive one full run through the HTTP API: health -> start run -> approve -> tracker.

    .venv/Scripts/python .claude/skills/run-job-copilot/smoke.py [base_url]

Exits non-zero on the first unexpected response. Stdlib only.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
RESUME = (
    "Asha Rao, Backend Engineer, Bengaluru. 4 years Python, FastAPI, PostgreSQL, Redis, "
    "Kubernetes. Built payment services handling 2 million UPI transactions a day."
)


def call(method: str, path: str, body: dict | None = None) -> dict | list:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as res:
            return json.loads(res.read())
    except urllib.error.HTTPError as exc:
        sys.exit(f"FAIL {method} {path}: {exc.code} {exc.read().decode()}")


def check(cond: bool, msg: str, payload: object) -> None:
    if not cond:
        sys.exit(f"FAIL {msg}: {json.dumps(payload, indent=2)[:2000]}")


health = call("GET", "/health")
print("health   :", health)

run = call("POST", "/runs", {"resume_text": RESUME,
                             "preferences": {"roles": ["backend engineer"], "locations": ["Bengaluru"]}})
check(run["status"] == "awaiting_review", "run should pause for review", run)
print("path     :", " -> ".join(run["path"]))
for s in run["scored"]:
    print(f"score    : {s['score']:3d} {s['job_id']} gaps={s['gaps']}")
drafts = run["review"]["drafts"]
print("drafts   :", [d["job_id"] for d in drafts])

done = call("POST", f"/runs/{run['thread_id']}/review",
            {"decisions": [{"job_id": d["job_id"], "decision": "approve"} for d in drafts]})
check(done["status"] == "completed", "run should complete after approval", done)
for job, path in done["artifacts"].items():
    print(f"artifact : {job} -> {path}")

apps = call("GET", "/applications")
print("tracker  :", apps)
print("OK")
