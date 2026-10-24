"""triage_submission wiring and OpenAI API hygiene (client config, request parameters)."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from conftest import base_extraction

import core
from core import EXTRACTION_SYSTEM_PROMPT, ExtractionError, PatientSubmission, TriageOutput, triage_submission
from extraction import PROMPT_CACHE_KEY, DocumentExtraction, ExtractionTelemetry, telemetry_from_response


def _usage(input_tokens=4000, cached=3500, output=600, reasoning=0):
    return SimpleNamespace(
        input_tokens=input_tokens, output_tokens=output,
        input_tokens_details=SimpleNamespace(cached_tokens=cached),
        output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
    )


def _client_returning(parsed, status: str = "completed", usage=None) -> Mock:
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(
        id="resp_test", model="gpt-5.6-luna", status=status, output_parsed=parsed, incomplete_details=None,
        usage=usage if usage is not None else _usage(),
    )
    return client


def test_request_uses_strict_schema_temperature_zero_and_no_storage(submission) -> None:
    client = _client_returning(DocumentExtraction(**base_extraction()))

    output = triage_submission(submission, model="gpt-5.6-luna", client=client)

    assert isinstance(output, TriageOutput)
    assert output.decision == "READY"
    assert client.responses.parse.call_count == 1
    call = client.responses.parse.call_args.kwargs
    assert call["model"] == "gpt-5.6-luna"
    assert call["instructions"] == EXTRACTION_SYSTEM_PROMPT
    assert call["text_format"] is DocumentExtraction
    assert call["temperature"] == 0
    assert call["reasoning"] == {"effort": "none"}
    assert call["store"] is False
    assert call["prompt_cache_key"] == PROMPT_CACHE_KEY


def test_structured_output_format_is_strict() -> None:
    parsing = pytest.importorskip("openai.lib._parsing._responses")
    text_format = parsing.type_to_text_format_param(DocumentExtraction)
    assert text_format["type"] == "json_schema"
    assert text_format["strict"] is True


def test_prompt_payload_contains_documents_only_no_patient_identifiers(submission) -> None:
    client = _client_returning(DocumentExtraction(**base_extraction()))
    triage_submission(PatientSubmission.model_validate(submission), model="m", client=client)

    payload = json.loads(client.responses.parse.call_args.kwargs["input"][0]["content"][0]["text"])
    assert set(payload) == {"documents", "requested_fallbacks"}
    assert payload["requested_fallbacks"] == {"procedure_date": False, "procedure_risk": False}
    assert [doc["doc_ref"] for doc in payload["documents"]] == ["D1", "D2", "D3"]
    assert {doc["document_class"] for doc in payload["documents"]} == {"HISTORY_AND_PHYSICAL", "SURGICAL_CONSENT", "OTHER"}
    serialized = json.dumps(payload)
    assert "p-1" not in serialized and "doc-hp" not in serialized


def test_fallbacks_are_requested_only_for_unusable_structured_fields(submission) -> None:
    submission["procedure"]["procedure_date"] = None
    submission["procedure"]["procedure_risk"] = "VERY HIGH"
    client = _client_returning(DocumentExtraction(**base_extraction()))
    triage_submission(submission, model="m", client=client)
    payload = json.loads(client.responses.parse.call_args.kwargs["input"][0]["content"][0]["text"])
    assert payload["requested_fallbacks"] == {"procedure_date": True, "procedure_risk": True}


def test_refusal_or_truncation_raises_instead_of_guessing(submission) -> None:
    with pytest.raises(ExtractionError):
        triage_submission(submission, model="m", client=_client_returning(None))
    with pytest.raises(ExtractionError):
        triage_submission(submission, model="m", client=_client_returning(DocumentExtraction(**base_extraction()), status="incomplete"))


def test_default_client_is_reused_with_timeout_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    constructed: list[dict] = []

    def fake_openai(**kwargs):
        constructed.append(kwargs)
        return object()

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=fake_openai))
    core.get_openai_client.cache_clear()
    try:
        first = core.get_openai_client()
        second = core.get_openai_client()
    finally:
        core.get_openai_client.cache_clear()

    assert first is second
    assert constructed == [{"timeout": 30.0, "max_retries": 2}]


# ---------------------------------------------------------------- prompt caching + telemetry


def test_static_prefix_and_cache_key_are_identical_across_cases(submission) -> None:
    import copy

    fallback_case = copy.deepcopy(submission)
    fallback_case["procedure"]["procedure_date"] = None
    fallback_case["procedure"]["procedure_risk"] = "unknown"
    calls = []
    for case in (submission, fallback_case):
        client = _client_returning(DocumentExtraction(**base_extraction()))
        triage_submission(case, model="m", client=client)
        calls.append(client.responses.parse.call_args.kwargs)

    for key in ("instructions", "text_format", "prompt_cache_key", "temperature", "reasoning", "model"):
        assert calls[0][key] == calls[1][key], key
    assert calls[0]["prompt_cache_key"] == PROMPT_CACHE_KEY
    # Case-specific flags are the tail of the only (user) message.
    for call in calls:
        text = call["input"][0]["content"][0]["text"]
        assert text.index('"documents"') < text.index('"requested_fallbacks"')
        assert text.rstrip("}").rstrip().endswith("false") or text.rstrip("}").rstrip().endswith("true")


def test_telemetry_reports_tokens_cache_hits_and_reaches_the_sink(submission) -> None:
    seen: list[ExtractionTelemetry] = []
    client = _client_returning(DocumentExtraction(**base_extraction()), usage=_usage(4000, 3000, 500, 7))
    output = triage_submission(submission, model="m", client=client, on_telemetry=seen.append)

    (telemetry,) = seen
    assert (telemetry.input_tokens, telemetry.cached_input_tokens, telemetry.output_tokens, telemetry.reasoning_tokens) == (4000, 3000, 500, 7)
    assert telemetry.cache_hit_ratio == 0.75
    assert telemetry.prompt_cache_key == PROMPT_CACHE_KEY
    assert "telemetry" not in output.model_dump_json() and "cached" not in output.model_dump_json()


def test_telemetry_tolerates_missing_usage_fields() -> None:
    sparse = SimpleNamespace(id=None, model=None, status="completed",
                             usage=SimpleNamespace(input_tokens=1200, output_tokens=None, input_tokens_details=None))
    telemetry = telemetry_from_response(sparse, latency_ms=10)
    assert (telemetry.input_tokens, telemetry.cached_input_tokens, telemetry.output_tokens) == (1200, 0, 0)
    assert telemetry.cache_hit_ratio == 0.0
    assert telemetry_from_response(SimpleNamespace(), latency_ms=0).input_tokens == 0


def test_failed_calls_still_report_telemetry(submission) -> None:
    seen: list[ExtractionTelemetry] = []
    with pytest.raises(ExtractionError):
        triage_submission(submission, model="m", client=_client_returning(None), on_telemetry=seen.append)
    assert len(seen) == 1 and seen[0].input_tokens == 4000
