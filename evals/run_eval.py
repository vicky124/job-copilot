"""Offline eval over the golden dataset. CI gate: fail if faithfulness drops below 0.95.

    python -m evals.run_eval                 # uses the configured provider profile
    LLM_PROVIDER_PROFILE=anthropic python -m evals.run_eval --variant variant-b

With provider_profile=fake (CI default) the run checks the harness itself; the
real comparison between prompt variants runs in staging against real models and
is pushed to Langfuse as a dataset experiment.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from copilot.adapters.llm import FakeLLMClient
from copilot.adapters.prompts import FilePromptStore
from copilot.domain.models import CandidateProfile, EvidencePair, JobPosting
from copilot.services.tailor import TailorService
from evals.scorers import ats_readability, faithfulness, keyword_coverage

ROOT = Path(__file__).resolve().parents[1]
FAITHFULNESS_GATE = 0.95


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "evals" / "datasets" / "golden.jsonl"))
    parser.add_argument("--variant", default="production")
    args = parser.parse_args(argv)

    rows = [json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line.strip()]
    prompts = FilePromptStore(ROOT / "prompts")
    scores: dict[str, list[float]] = {"faithfulness": [], "keyword_coverage": [], "ats": []}
    for row in rows:
        profile = CandidateProfile.model_validate(row["profile"])
        job = JobPosting.model_validate(row["job"])
        # Fake model replays the reference output; swap for build_llm_client in staging.
        llm = FakeLLMClient(default=json.dumps(row["reference"]))
        tailored = TailorService(llm, prompts).tailor(
            profile,
            job,
            [EvidencePair(requirement=r, evidence=None) for r in job.required_skills],
            prompt_variant=args.variant,
        )
        scores["faithfulness"].append(faithfulness(profile, tailored))
        scores["keyword_coverage"].append(keyword_coverage(job.required_skills, tailored))
        scores["ats"].append(ats_readability(tailored))

    summary = {k: round(statistics.mean(v), 3) for k, v in scores.items()}
    print(json.dumps({"variant": args.variant, "n": len(rows), **summary}, indent=2))
    if summary["faithfulness"] < FAITHFULNESS_GATE:
        print(f"FAIL: faithfulness {summary['faithfulness']} < {FAITHFULNESS_GATE}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
