"""Record real model extractions for selected cases as offline regression fixtures.

Usage (needs OPENAI_API_KEY):
    uv run --with 'openai>=2.0.0' --with 'pydantic>=2.8.0' python tests/fixtures/record_extractions.py 16 25

Each fixture stores the raw DocumentExtraction plus the prompt version it came from;
tests replay it through verification + policy without network access.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core import get_openai_client  # noqa: E402
from extraction import EXTRACTION_PROMPT_VERSION, call_extractor, preprocess  # noqa: E402
from schemas import PatientSubmission  # noqa: E402

MODEL = "gpt-5.6-luna"


def main(indices: list[int]) -> None:
    cases = [json.loads(line) for line in (ROOT / "data" / "patients_sample_50.jsonl").read_text().splitlines() if line.strip()]
    for index in indices:
        case = cases[index]
        facts = preprocess(PatientSubmission.model_validate(case["submission"]), model=MODEL)
        extraction, telemetry = call_extractor(get_openai_client(), model=MODEL, facts=facts)
        path = Path(__file__).parent / f"extraction_{case['case_id']}.json"
        path.write_text(json.dumps({
            "case_id": case["case_id"],
            "model": MODEL,
            "prompt_version": EXTRACTION_PROMPT_VERSION,
            "extraction": extraction.model_dump(),
        }, indent=2) + "\n")
        print(f"wrote {path.name} ({telemetry.input_tokens if telemetry else 0} input tokens)")


if __name__ == "__main__":
    main([int(arg) for arg in sys.argv[1:]])
