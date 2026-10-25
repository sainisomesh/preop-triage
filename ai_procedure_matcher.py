"""Optional AI suggestion for procedure names that fail deterministic matching (off by default).

    from ai_procedure_matcher import openai_procedure_matcher
    triage_submission(case, model=MODEL, procedure_matcher=openai_procedure_matcher(MODEL))

Called only for signed consents whose procedure name failed reference.procedure_match_detail
(synonyms the table doesn't know yet, e.g. "ACL reconstruction" vs "anterior cruciate
ligament repair"). The answer is written to audit.informational_checks for the nurse and never
changes the consent rule, which has already failed closed. Pairs a reviewer confirms belong in
reference.PROCEDURE_SYNONYMS, so the deterministic path learns them.

This is a separate request from document extraction, so the extraction prompt, schema and
cached prefix are unchanged.
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, ConfigDict

PROMPT = """
You compare two surgical procedure names: the procedure scheduled and the procedure named on a signed consent form.
Answer whether they describe the same operation on the same anatomy. Treat abbreviations, synonyms and
open/laparoscopic wording carefully: different anatomy, a different side (left/right) or a different operation
means NOT the same. When unsure, answer false. Give a one-sentence rationale.
""".strip()


class ProcedureEquivalence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    same_procedure: bool
    rationale: str


def openai_procedure_matcher(model: str, client: Any | None = None) -> Callable[[str, str], tuple[bool, str, str]]:
    def match(scheduled: str, consented: str) -> tuple[bool, str, str]:
        from core import get_openai_client  # lazy: keeps this module importable offline

        api = client or get_openai_client()
        kwargs: dict[str, Any] = dict(
            model=model,
            instructions=PROMPT,
            input=[{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": f"Scheduled: {scheduled}\nConsent: {consented}"}]}],
            text_format=ProcedureEquivalence,
            temperature=0,
            store=False,
            max_output_tokens=200,
        )
        if model.startswith("gpt-5"):
            kwargs["reasoning"] = {"effort": "none"}  # gpt-5.x only accepts temperature with reasoning off
        response = api.responses.parse(**kwargs)
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("procedure matcher returned no parsed output")
        return parsed.same_procedure, parsed.rationale, f"{model} procedure matcher"

    return match
