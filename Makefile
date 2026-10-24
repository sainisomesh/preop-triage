# Load local secrets (e.g. OPENAI_API_KEY) if present; exported to recipes.
-include .env
export

INPUT ?= data/patients_sample_50.jsonl
OUTPUT ?= data/baseline_outputs.jsonl
REPORT ?= data/eval_report.json
DETERMINISM_REPORT ?= data/determinism_report.json
CONSISTENCY_REPORT ?= data/consistency_report.json
RUNS ?= 3
MODEL ?= gpt-5.6-luna

.PHONY: baseline evals determinism consistency score report test all clean

baseline:
	uv run run_baseline.py \
		--input $(INPUT) \
		--output $(OUTPUT) \
		--model $(MODEL)

evals:
	uv run run_evals.py \
		--input $(INPUT) \
		--outputs $(OUTPUT) \
		--report $(REPORT)

determinism:
	uv run run_evals.py \
		--determinism \
		--input $(INPUT) \
		--model $(MODEL) \
		--report $(DETERMINISM_REPORT)

consistency:
	uv run run_consistency.py \
		--input $(INPUT) \
		--model $(MODEL) \
		--runs $(RUNS) \
		--report $(CONSISTENCY_REPORT)

score:
	@python3 -c 'import json; r=json.load(open("$(REPORT)")); s=(r.get("primary_score",{}) or {}).get("value_pct"); print(s if s is not None else r.get("local_metrics_summary",{}).get("aggregate_local_score_pct", 0.0))'

report:
	uv run view_report.py --report $(REPORT)

test:
	uv run python -m pytest tests

all: baseline evals determinism score

clean:
	rm -f data/baseline_outputs.jsonl \
		data/eval_report.json \
		data/determinism_report.json \
		data/consistency_report.json
