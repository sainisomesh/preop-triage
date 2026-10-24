"""Submission input, triage output and audit-trail models (scored fields: decision, issues[].category)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from reference import ProcedureRisk

Decision = Literal["READY", "NEEDS_FOLLOW_UP", "NOT_CLEARED"]
IssueCategory = Literal[
    "REQUIRED_DOCUMENTATION",
    "REQUIRED_TESTING",
    "ANTICOAGULATION_MANAGEMENT",
    "ACUTE_SAFETY_EXCLUSION",
    "MISSING_REQUIRED_DATA",
]
RuleStatus = Literal["PASS", "FAIL", "INDETERMINATE", "NOT_APPLICABLE"]

__all__ = [
    "Decision",
    "IssueCategory",
    "ProcedureRisk",
    "RuleStatus",
]

# -------------------------
# Input
# -------------------------


class PatientName(BaseModel):
    given: str | None = None
    family: str | None = None


class PatientInfo(BaseModel):
    id: str | None = None
    mrn: str | None = None
    name: PatientName | None = None
    dob: str | None = None
    sex: str | None = None


class ProcedureInfo(BaseModel):
    case_id: str | None = None
    procedure_type: str | None = None
    # Raw strings on purpose: a malformed risk or date must reach the pipeline so it
    # can be flagged and reconciled, rather than crash validation upstream.
    procedure_risk: str | None = None
    procedure_date: str | None = None
    is_elective: bool | None = None
    location: str | None = None


class BloodPressureVital(BaseModel):
    type: str | None = None
    systolic: float | int | None = None
    diastolic: float | int | None = None
    date: str | None = None
    source: str | None = None


class TemperatureVital(BaseModel):
    type: str | None = None
    value_f: float | int | None = None
    date: str | None = None
    source: str | None = None


class GenericVital(BaseModel):
    type: str | None = None
    date: str | None = None
    source: str | None = None


Vital = BloodPressureVital | TemperatureVital | GenericVital


class LabResult(BaseModel):
    id: str | None = None
    code: str | None = None
    display: str | None = None
    effective_at: str | None = None
    status: str | None = None
    source: str | None = None


class Medication(BaseModel):
    name: str | None = None
    active: bool | None = None


class Condition(BaseModel):
    name: str | None = None
    active: bool | None = None


class Document(BaseModel):
    doc_id: str | None = None
    type: str | None = None
    date: str | None = None
    author: str | None = None
    text: str | None = None


class SubmissionMetadata(BaseModel):
    submission_received_at: str | None = None
    source_system: str | None = None


class PatientSubmission(BaseModel):
    patient: PatientInfo | None = None
    procedure: ProcedureInfo | None = None
    vitals: list[Vital] = Field(default_factory=list)
    labs: list[LabResult] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    documents: list[Document] = Field(default_factory=list)
    metadata: SubmissionMetadata | None = None


# -------------------------
# Output
# -------------------------

SourceType = Literal["document", "lab", "vital", "medication", "procedure", "metadata"]


class EvidenceQuote(BaseModel):
    # verbatim=True: text is a source document line, verified by extraction.locate_quote.
    # verbatim=False: a rendering of structured fields (lab row, vital, med-list entry).

    source_type: SourceType
    source_id: str
    date: str | None
    text: str
    verbatim: bool


class Evidence(BaseModel):
    rule: str
    summary: str
    source_ids: list[str]
    quotes: list[EvidenceQuote]


class TriageIssue(BaseModel):
    category: IssueCategory
    evidence: Evidence


class RuleCheck(BaseModel):
    rule: str
    status: RuleStatus
    issue_category: IssueCategory | None
    evidence: Evidence


class ResolvedProcedureField(BaseModel):
    field: Literal["procedure_date", "procedure_risk"]
    value: str | None
    source: Literal["STRUCTURED", "DOCUMENT", "UNRESOLVED"]
    structured_raw: str | None
    evidence: list[EvidenceQuote]
    notes: list[str]


class ExcludedEvidence(BaseModel):
    source_type: SourceType
    source_id: str
    reason: str


class InformationalCheck(BaseModel):
    # Care-team context (e.g. NPO review). Kept outside rule_checks so it cannot reach decide().

    check: str
    status: str
    summary: str
    quotes: list[EvidenceQuote]
    affects_decision: Literal[False] = False


class AuditTrail(BaseModel):
    policy_version: str
    pipeline_version: str
    extraction_model: str
    facility_timezone: str | None
    review_time: str | None
    procedure_type: str | None
    procedure_date: ResolvedProcedureField
    procedure_risk: ResolvedProcedureField
    rule_checks: list[RuleCheck]
    excluded_evidence: list[ExcludedEvidence]
    informational_checks: list[InformationalCheck]
    warnings: list[str]


class TriageOutput(BaseModel):
    decision: Decision
    issues: list[TriageIssue]
    explanation: str
    audit: AuditTrail | None = None
