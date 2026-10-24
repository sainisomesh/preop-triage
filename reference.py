"""Reference data and pure normalizers shared by extraction.py (layer 1) and policy_engine.py (layer 2)."""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from decimal import Decimal
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

POLICY_VERSION = "cadence-preop-2026-01-01"

# -------------------------
# Policy thresholds
# -------------------------

HP_WINDOW_DAYS = 30
CBC_WINDOW_DAYS_STANDARD = 30  # LOW / MODERATE risk
HIGH_RISK_LAB_WINDOW_DAYS = 14  # CBC and CMP for HIGH risk
SYSTOLIC_EXCLUSION_MMHG = 180  # >= excludes
DIASTOLIC_EXCLUSION_MMHG = 110  # >= excludes
TEMPERATURE_EXCLUSION_F = Decimal("100.4")  # > excludes

ProcedureRisk = Literal["LOW", "MODERATE", "HIGH"]
PROCEDURE_RISKS: tuple[ProcedureRisk, ...] = ("LOW", "MODERATE", "HIGH")

# -------------------------
# Medications
# -------------------------
# Generic -> aliases (brand names, abbreviations), from FDA brand/generic pairs via GoodRx,
# RxList, DrugBank and IHS/NV Medicaid formulary briefs. Antiplatelets are listed so Rule 3
# can recognise and explicitly exclude them. Add new agents here; nothing else changes.

ANTICOAGULANTS: dict[str, tuple[str, ...]] = {
    # Vitamin K antagonists
    "warfarin": ("warfarin", "coumadin", "jantoven"),
    "acenocoumarol": ("acenocoumarol", "sintrom"),
    "phenprocoumon": ("phenprocoumon", "marcumar"),
    # Direct oral anticoagulants
    "apixaban": ("apixaban", "eliquis"),
    "rivaroxaban": ("rivaroxaban", "xarelto"),
    "dabigatran": ("dabigatran", "pradaxa"),
    "edoxaban": ("edoxaban", "savaysa", "lixiana"),
    "betrixaban": ("betrixaban", "bevyxxa"),
    # Parenteral
    "heparin": ("heparin", "unfractionated heparin", "ufh"),
    "enoxaparin": ("enoxaparin", "lovenox"),
    "dalteparin": ("dalteparin", "fragmin"),
    "tinzaparin": ("tinzaparin", "innohep"),
    "fondaparinux": ("fondaparinux", "arixtra"),
    "argatroban": ("argatroban",),
    "bivalirudin": ("bivalirudin", "angiomax"),
    "desirudin": ("desirudin", "iprivask"),
}

ANTIPLATELETS: dict[str, tuple[str, ...]] = {
    "aspirin": ("aspirin", "asa", "ecotrin", "bayer aspirin"),
    "clopidogrel": ("clopidogrel", "plavix"),
    "prasugrel": ("prasugrel", "effient"),
    "ticagrelor": ("ticagrelor", "brilinta"),
    "ticlopidine": ("ticlopidine", "ticlid"),
    "cilostazol": ("cilostazol", "pletal"),
    "dipyridamole": ("dipyridamole", "persantine"),
    "aspirin/dipyridamole": ("aggrenox",),
    "vorapaxar": ("vorapaxar", "zontivity"),
    "cangrelor": ("cangrelor", "kengreal"),
}

DrugClass = Literal["ANTICOAGULANT", "ANTIPLATELET"]


def _normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", value).strip()


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _normalize_text(value))


def _build_alias_index() -> list[tuple[tuple[str, ...], str, DrugClass]]:
    index: list[tuple[tuple[str, ...], str, DrugClass]] = []
    for table, drug_class in ((ANTICOAGULANTS, "ANTICOAGULANT"), (ANTIPLATELETS, "ANTIPLATELET")):
        for generic, aliases in table.items():
            for alias in aliases:
                index.append((tuple(_tokens(alias)), generic, drug_class))
    # Longest first so multi-word aliases ("unfractionated heparin") are tried before "heparin".
    index.sort(key=lambda item: (-len(item[0]), item[0]))
    return index


_ALIAS_INDEX = _build_alias_index()


def classify_drug(name: str) -> list[tuple[str, DrugClass]]:
    # Whole-token matching so "asa" never matches inside another word. Only run on
    # medication-list names and extracted drug names, never on free text, where
    # "ASA physical status" would read as aspirin.

    tokens = _tokens(name)
    found: set[tuple[str, DrugClass]] = set()
    for alias_tokens, generic, drug_class in _ALIAS_INDEX:
        width = len(alias_tokens)
        for start in range(len(tokens) - width + 1):
            if tuple(tokens[start : start + width]) == alias_tokens:
                found.add((generic, drug_class))
                break
    return sorted(found)


def anticoagulant_generics(name: str) -> list[str]:
    return [generic for generic, drug_class in classify_drug(name) if drug_class == "ANTICOAGULANT"]


# -------------------------
# Labs
# -------------------------

LabType = Literal["CBC", "CMP", "BMP"]
# FHIR Observation statuses for a released result. amended/corrected replace an earlier
# final; registered, preliminary, cancelled, entered-in-error and unknown are excluded.
ACCEPTED_LAB_STATUSES = frozenset({"final", "amended", "corrected"})

_LAB_CODE_MAP: dict[str, LabType] = {"CBC": "CBC", "CMP": "CMP", "BMP": "BMP"}
_LAB_DISPLAY_PATTERNS: tuple[tuple[re.Pattern[str], LabType], ...] = (
    (re.compile(r"\bcomplete blood count\b|\bhemogram\b|\bcbc\b"), "CBC"),
    (re.compile(r"\bcomprehensive metabolic panel\b|\bcmp\b"), "CMP"),
    (re.compile(r"\bbasic metabolic panel\b|\bbmp\b"), "BMP"),
)


def normalize_lab(code: str | None, display: str | None) -> tuple[LabType | None, str]:
    # Code wins ("LAB-CBC" -> CBC), display is the fallback. A BMP is never treated as a
    # CMP, and a code/display conflict returns None so preprocessing logs it for review.

    from_code: LabType | None = None
    if code:
        cleaned = re.sub(r"^(LAB|LB|L)[-_ ]+", "", code.strip().upper())
        from_code = _LAB_CODE_MAP.get(cleaned)

    from_display: LabType | None = None
    if display:
        normalized = _normalize_text(display)
        for pattern, lab_type in _LAB_DISPLAY_PATTERNS:
            if pattern.search(normalized):
                from_display = lab_type
                break

    if from_code and from_display and from_code != from_display:
        return None, f"code {code!r} ({from_code}) conflicts with display {display!r} ({from_display})"
    if from_code:
        return from_code, f"code {code!r} -> {from_code}"
    if from_display:
        return from_display, f"display {display!r} -> {from_display}"
    return None, "not a CBC/CMP/BMP"


# -------------------------
# Document titles
# -------------------------

DocumentClass = Literal["HISTORY_AND_PHYSICAL", "SURGICAL_CONSENT", "OTHER"]


def _edit_distance(a: str, b: str) -> int:
    # Optimal string alignment: a swapped pair ("Pyhsical") costs 1, not 2.

    rows = len(a) + 1
    cols = len(b) + 1
    dist = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dist[i][0] = i
    for j in range(cols):
        dist[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dist[i][j] = min(dist[i - 1][j] + 1, dist[i][j - 1] + 1, dist[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                dist[i][j] = min(dist[i][j], dist[i - 2][j - 2] + 1)
    return dist[-1][-1]


def _has_fuzzy_token(tokens: list[str], target: str, max_distance: int) -> bool:
    return any(
        abs(len(token) - len(target)) <= max_distance and _edit_distance(token, target) <= max_distance
        for token in tokens
    )


def classify_document_title(title: str | None) -> tuple[DocumentClass, str]:
    # Title-only and typo-tolerant. The class is sent to the extractor so it reports
    # consent/H&P findings for exactly these documents; Rule 1 only considers them.

    if not title:
        return "OTHER", "no title"
    normalized = _normalize_text(title).replace("&", " and ")
    tokens = re.findall(r"[a-z0-9]+", normalized)
    joined = " ".join(tokens)

    is_hp = bool(re.search(r"\bh and p\b|\bhandp\b|\bhp\b", joined)) or (
        _has_fuzzy_token(tokens, "history", 2) and _has_fuzzy_token(tokens, "physical", 2)
    )
    is_consent = _has_fuzzy_token(tokens, "consent", 1)

    if is_hp and is_consent:
        return "OTHER", f"title {title!r} matches both H&P and consent patterns"
    if is_hp:
        return "HISTORY_AND_PHYSICAL", f"title {title!r} matches H&P pattern"
    if is_consent:
        return "SURGICAL_CONSENT", f"title {title!r} matches consent pattern"
    return "OTHER", f"title {title!r} is not an H&P or consent title"


# Nursing intake / pre-admission notes; used only by the informational NPO check.
_NURSING_INTAKE_TITLE = re.compile(r"\bnurs\w*|\bintake\b|\bpre ?admission testing\b|\bpat visit\b")


def is_nursing_intake_title(title: str | None) -> bool:
    if not title:
        return False
    normalized = " ".join(re.findall(r"[a-z0-9]+", _normalize_text(title)))
    return bool(_NURSING_INTAKE_TITLE.search(normalized))


NPO_PATTERN = re.compile(
    r"\bNPO\b|\bn\.p\.o\.|nothing by mouth|nil per os|\bfasting\b|nothing to eat|nothing to drink|clear liquids",
    re.IGNORECASE,
)
NPO_NOT_REVIEWED_PATTERN = re.compile(
    r"\bnot\b[^.;]{0,30}\b(reviewed|given|discussed|provided|instructed|completed|done)\b|\bdeferred\b|\bpending\b"
    r"|\bunable to\b|\bdeclined\b|\bnot yet\b|\bto be reviewed\b",
    re.IGNORECASE,
)
NPO_ACKNOWLEDGED_PATTERN = re.compile(
    r"\breviewed\b|verbali[sz]ed understanding|verbali[sz]es understanding|\bteach-?back\b|\bdiscussed\b|\beducated\b"
    r"|\bunderstands\b|\backnowledged\b",
    re.IGNORECASE,
)


# -------------------------
# Dates, times and quotes
# -------------------------

_DATE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), "ymd"),
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"), "mdy"),
)


def parse_iso_date(value: str | None) -> date | None:

    if not value or not isinstance(value, str):
        return None
    try:
        if "T" in value:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


TimeBasis = Literal["LOCAL", "UTC_OFFSET", "UNSPECIFIED", "DATE_ONLY"]


def load_timezone(name: str | None) -> ZoneInfo | None:
    # IANA name such as "America/Chicago"; a bad name fails before any API call.

    if not name:
        return None
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"unknown facility timezone {name!r}") from exc


def to_facility_time(value: str | None, tz: ZoneInfo | None) -> tuple[date | None, int | None, TimeBasis]:
    # Returns (date, minutes, basis). With a facility tz, "...Z" timestamps become local
    # wall-clock (LOCAL) so they can be ordered against note times; without one they stay
    # UTC_OFFSET and policy_engine.most_recent_reading refuses to order them against notes.
    # The converted date also feeds lab windows, so a 03:00Z result can land on the prior day.

    if not value or not isinstance(value, str) or not value.strip():
        return None, None, "UNSPECIFIED"
    text = value.strip()
    if "T" not in text and " " not in text:
        return parse_iso_date(text), None, "DATE_ONLY"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None, None, "UNSPECIFIED"
    if parsed.tzinfo is not None:
        if tz is not None:
            parsed, basis = parsed.astimezone(tz), "LOCAL"
        else:
            basis = "UTC_OFFSET"
    else:
        basis = "LOCAL" if tz is not None else "UNSPECIFIED"
    return parsed.date(), parsed.hour * 60 + parsed.minute, basis


# Shared by extraction (decides whether to request a document fallback) and the policy
# engine (decides which value wins), so both layers agree on what "usable" means.
def structured_procedure_date(raw: object) -> date | None:

    return parse_iso_date(raw) if isinstance(raw, str) and raw.strip() else None


def structured_procedure_risk(raw: object) -> ProcedureRisk | None:

    if not isinstance(raw, str):
        return None
    normalized = raw.strip().upper()
    return normalized if normalized in PROCEDURE_RISKS else None  # type: ignore[return-value]


def find_dates(text: str) -> list[date]:
    # US M/D/YYYY and ISO only; impossible dates (02/30) are skipped, not guessed.

    found: list[date] = []
    for pattern, order in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            parts = [int(part) for part in match.groups()]
            year, month, day = parts if order == "ymd" else (parts[2], parts[0], parts[1])
            try:
                found.append(date(year, month, day))
            except ValueError:
                continue
    return sorted(set(found))


def parse_clock_time(value: str | None) -> int | None:

    if not value:
        return None
    match = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", value)
    if not match:
        return None
    hours, minutes = int(match.group(1)), int(match.group(2))
    if hours > 23 or minutes > 59:
        return None
    return hours * 60 + minutes


# Degree-sign variants seen in model output: gpt-4.1-mini emitted U+0017 + "6" for "°" in JSON
# strings; escaped and look-alike forms are folded too. The sign is then optional for matching.
_DEGREE_VARIANTS = re.compile(r"\x17\s?6|\\u00b0|\\xb0|[°º˚∘]", re.IGNORECASE)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def fold_degree_signs(value: str) -> str:
    # Canonical "°" before and after NFKC: NFKC turns "º" into "o" but "℃" into "°C".
    value = _DEGREE_VARIANTS.sub("°", value)
    return _DEGREE_VARIANTS.sub("°", unicodedata.normalize("NFKC", value))


def normalize_for_match(value: str) -> str:
    # Used only to *compare* model quotes with source text; evidence keeps the source line.
    value = fold_degree_signs(value)
    value = _CONTROL_CHARS.sub("", value)
    value = re.sub(r"\s*°\s*", " ", value)
    value = value.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    value = value.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", value).strip().casefold()


def locate_quote(text: str, quote: str) -> tuple[int, str] | None:
    # The anti-hallucination check: a quote that is not in the source returns None and the
    # fact is rejected. Matches are snapped to the whole source line(s), so evidence text is
    # the same across runs even when the model quotes a slightly different span.

    needle = normalize_for_match(quote)
    if not needle:
        return None
    lines = text.splitlines()
    for width in (1, 2, 3):
        for start in range(len(lines) - width + 1):
            window = " ".join(lines[start : start + width])
            if needle in normalize_for_match(window):
                return start, "\n".join(line.strip() for line in lines[start : start + width])
    return None


# Rule 3: a hold/resume instruction counts only with a concrete time ("2 days before",
# "POD 1"); "per surgical team" or "as directed" does not.
TIMING_PATTERN = re.compile(
    r"\d|\bday\b|\bdays\b|\bhour|\bweek|\bmorning\b|\bevening\b|\bnight\b|\btonight\b|\btomorrow\b"
    r"|\bpod\b|\bpostoperative day\b|\bpost-op day\b|\bprior to\b|\bbefore surgery\b",
    re.IGNORECASE,
)

# Safety net over the extractor: a "signed" consent whose signature line reads like this is downgraded.
UNSIGNED_CONSENT_PATTERN = re.compile(
    r"pending|not completed|not yet signed|unsigned|not signed|_{3,}|will return|to be signed|awaiting",
    re.IGNORECASE,
)

# Risk statements that are not the procedure risk category (bleeding, fall, ASA class, ...).
PROCEDURE_RISK_EXCLUDE_PATTERN = re.compile(
    r"bleed|thrombo|cardiac|fall|anesthe|\basa\b|stroke|aspiration", re.IGNORECASE
)

_PROCEDURE_STOPWORDS = frozenset({"elective", "procedure", "surgery", "surgical", "the", "of", "a", "an", "and", "with", "for", "left", "right", "bilateral"})


def procedure_tokens(value: str) -> frozenset[str]:
    return frozenset(token for token in _tokens(value) if token not in _PROCEDURE_STOPWORDS)


def procedures_match(scheduled: str, consented: str) -> bool:
    # Every content word of the shorter name must appear in the longer one. Strict on
    # purpose: "...bypass grafting" vs "gastric bypass" fails, and synonyms ("hip
    # replacement" vs "arthroplasty") fail closed into REQUIRED_DOCUMENTATION follow-up.

    scheduled_tokens = procedure_tokens(scheduled)
    consented_tokens = procedure_tokens(consented)
    if not scheduled_tokens or not consented_tokens:
        return False
    smaller, larger = sorted((scheduled_tokens, consented_tokens), key=len)
    return smaller <= larger
