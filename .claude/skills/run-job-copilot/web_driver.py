"""Drive the Next.js UI headless: paste resume -> Find and tailor -> Approve all.
Screenshots land in outputs/screens/. Uses the system Edge (channel="msedge"),
so no `playwright install` browser download is needed on Windows.

    .venv/Scripts/python .claude/skills/run-job-copilot/web_driver.py [web_url]

Needs the API (serve_offline.py) on :8000 and `npm run dev` on :3000.
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:3000"
OUT = Path(__file__).resolve().parents[3] / "outputs" / "screens"
RESUME = (
    "Asha Rao, Backend Engineer, Bengaluru. 4 years Python, FastAPI, PostgreSQL, Redis, "
    "Kubernetes. Built payment services handling 2 million UPI transactions a day."
)

OUT.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    page = browser.new_page(viewport={"width": 1100, "height": 900})
    page.on("console", lambda m: m.type == "error" and print("console:", m.text))
    # First hit compiles the page in next dev; give it time.
    page.goto(URL, timeout=120_000)
    page.get_by_label("Resume text").fill(RESUME)
    page.screenshot(path=OUT / "1-filled.png", full_page=True)

    page.get_by_role("button", name="Find and tailor").click()
    expect(page.get_by_role("heading", name="Review drafts")).to_be_visible(timeout=120_000)
    page.screenshot(path=OUT / "2-review.png", full_page=True)

    page.get_by_role("button", name="Approve all").click()
    expect(page.get_by_role("heading", name="Review drafts")).to_be_hidden(timeout=120_000)
    page.screenshot(path=OUT / "3-approved.png", full_page=True)
    print(page.locator("body").inner_text())
    browser.close()
print("screens ->", OUT)
