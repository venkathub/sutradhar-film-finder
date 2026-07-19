"""LLM candidate-edge extraction (P1 task 11, DEC-P1-4 — the GPU-session job).

An LLM reads Wikipedia lead/plot prose (the revision-pinned ``plot_texts`` corpus) and
*proposes* remake/dub/sequel/based_on edges as structured rows in ``candidate_edges`` —
**never** into ``edges`` directly (the human gate, task 12, promotes).

Honesty contract:
- Model output is validated against a pydantic schema; malformed output is **dropped and
  counted, never repaired** into the DB.
- The ``supporting_sentence`` must appear verbatim (whitespace-normalized) in the source
  text, or the proposal is dropped as *unsupported* — a hallucinated citation never lands.
- Every candidate carries the page + revision pin, the model id, and an ``extraction_run``
  hash (prompt template + model + page revisions) — the reproducibility stamp.
- Title→version resolution is conservative: only an unambiguous ≥0.9 ``resolve_title`` hit
  binds a version_id; everything else keeps raw strings for the human reviewer.

Compute placement: the batch runs against the env-driven ``LLM_BASE_URL`` (the ephemeral
A100 vLLM endpoint, DEC-0003) via the P0 ``LLMClient`` — this module does not know or care
that the endpoint is JarvisLabs. CI never calls a model: it replays a recorded artifact.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from sutradhar.graph.repository import resolve_title
from sutradhar.graph.schema import CandidateEdge, PlotText, Version
from sutradhar.serving.llm_client import LLMClient

PROMPT_TEMPLATE = """\
You are a precise information-extraction system for Indian cinema.

From the Wikipedia article excerpt below, extract every film-to-film relationship that the \
text EXPLICITLY states. Do not infer or guess relationships that are not stated.

Relationship types (exact strings):
- "is_remake_of": SRC is a remake of DST (a new film re-telling DST's story, different cast).
- "is_official_dub_of": SRC is a dubbed version of DST (same film, re-recorded audio).
- "is_sequel_of": SRC is a sequel to DST.
- "based_on": SRC is adapted from a literary work DST (novel, novella, play).

Rules:
- src_title / dst_title: film or literary-work titles exactly as written in the text.
- src_language / dst_language: BCP-47-ish code (ml, ta, te, hi, kn, bn, si, zh, en) when the
  text states the language, else null.
- supporting_sentence: copy the single sentence that states the relationship VERBATIM.
- confidence: your confidence 0.0-1.0 that the text states this relationship.
- Output STRICT JSON, nothing else, matching:
  {{"relationships": [{{"edge_type": "...", "src_title": "...", "src_language": null,
  "dst_title": "...", "dst_language": null, "supporting_sentence": "...", "confidence": 0.9}}]}}
- If the text states no relationship, output {{"relationships": []}}.

ARTICLE ({page_title}):
{text}
"""

EdgeTypeLiteral = Literal["is_remake_of", "is_official_dub_of", "is_sequel_of", "based_on"]

RESOLVE_BIND_THRESHOLD = 0.9  # bind a version_id only at/above this resolve_title score


class ProposedEdge(BaseModel):
    model_config = ConfigDict(extra="ignore")

    edge_type: EdgeTypeLiteral
    src_title: str = Field(min_length=1)
    src_language: str | None = None
    dst_title: str = Field(min_length=1)
    dst_language: str | None = None
    supporting_sentence: str = Field(min_length=10)
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    relationships: list[ProposedEdge]


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _normalize_ws(text: str) -> str:
    return " ".join(text.split())


def parse_extraction_output(raw: str) -> ExtractionResponse | None:
    """Strict parse of one model response. ``None`` = malformed (dropped + counted).

    Tolerates trailing junk after ONE well-formed JSON object (guided decoding can emit
    continuation noise): the prefix object is taken as-is — content is never repaired.
    """
    cleaned = _FENCE_RE.sub("", raw.strip())
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            payload, _end = json.JSONDecoder().raw_decode(cleaned)
        except json.JSONDecodeError:
            return None
    try:
        return ExtractionResponse.model_validate(payload)
    except ValidationError:
        return None


def is_supported(sentence: str, source_text: str) -> bool:
    """The verbatim-evidence guard: the sentence must occur in the source text."""
    return _normalize_ws(sentence) in _normalize_ws(source_text)


def extraction_run_hash(model_id: str, page_revisions: dict[str, str]) -> str:
    """Reproducibility stamp: prompt template + model + exact page revisions."""
    blob = json.dumps(
        {
            "prompt_sha": hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest(),
            "model": model_id,
            "revisions": dict(sorted(page_revisions.items())),
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# --- Batch over the plot corpus ---


@dataclass
class ExtractionReport:
    pages_processed: int = 0
    responses_malformed: int = 0  # dropped, never repaired
    proposals_raw: int = 0
    proposals_unsupported: int = 0  # supporting sentence not verbatim in source → dropped
    proposals_duplicate: int = 0
    candidates_written: int = 0
    resolved_both_ends: int = 0
    kept_raw_titles: int = 0
    parse_failure_rate: float = 0.0
    run_hash: str = ""
    errors: list[str] = field(default_factory=list)


def run_extraction(
    client: LLMClient,
    pages: dict[str, dict[str, Any]],
) -> tuple[dict[str, str], ExtractionReport]:
    """Call the model once per page. Returns ``{page_key: raw_response}`` + a partial report.

    ``pages``: ``{page_key: {"title", "text", "revision"}}`` — the persisted artifact input.
    Network/model errors per page are recorded and skipped (the batch survives).
    """
    report = ExtractionReport()
    raw_outputs: dict[str, str] = {}
    guided = {"guided_json": ExtractionResponse.model_json_schema()}
    for key, page in sorted(pages.items()):
        prompt = PROMPT_TEMPLATE.format(page_title=page["title"], text=page["text"])
        try:
            raw = client.complete(prompt, max_tokens=6144, temperature=0.0, extra_body=guided)
        except Exception as exc:  # noqa: BLE001 — batch resilience; recorded, not hidden
            report.errors.append(f"{key}: {type(exc).__name__}: {exc}")
            continue
        raw_outputs[key] = raw
        report.pages_processed += 1
    return raw_outputs, report


def load_candidates(
    session: Session,
    raw_outputs: dict[str, str],
    pages: dict[str, dict[str, Any]],
    model_id: str,
    report: ExtractionReport | None = None,
) -> ExtractionReport:
    """Parse recorded outputs → validated proposals → ``candidate_edges`` rows. Idempotent."""
    report = report or ExtractionReport(pages_processed=len(raw_outputs))
    revisions = {k: str(p["revision"]) for k, p in pages.items()}
    report.run_hash = extraction_run_hash(model_id, revisions)

    # P7 task 7 (DEC-P7-1 finding 9): batch-local dedup. The SELECT below cannot
    # see pending (unflushed) inserts under autoflush=False, so duplicates WITHIN
    # one batch used to slip through — exposed the moment the DB-owned
    # uq_candidate_edges_dedup constraint landed. Key mirrors that constraint.
    seen_in_batch: set[tuple[str, str | None, str | None, str]] = set()

    for key, raw in sorted(raw_outputs.items()):
        page = pages[key]
        response = parse_extraction_output(raw)
        if response is None:
            report.responses_malformed += 1
            continue
        for proposal in response.relationships:
            report.proposals_raw += 1
            if not is_supported(proposal.supporting_sentence, str(page["text"])):
                report.proposals_unsupported += 1
                continue
            src_id = _bind_version(session, proposal.src_title, proposal.src_language)
            dst_id = _bind_version(session, proposal.dst_title, proposal.dst_language)
            if src_id is not None and src_id == dst_id:
                dst_id = None  # never propose a self-edge binding; keep raw for the reviewer
            dedup_key = (
                str(proposal.edge_type),
                proposal.src_title,
                proposal.dst_title,
                str(page["title"]),
            )
            duplicate = session.scalars(
                select(CandidateEdge).where(
                    CandidateEdge.edge_type == proposal.edge_type,
                    CandidateEdge.src_title_raw == proposal.src_title,
                    CandidateEdge.dst_title_raw == proposal.dst_title,
                    CandidateEdge.source_page == str(page["title"]),
                )
            ).first()
            if duplicate is not None or dedup_key in seen_in_batch:
                report.proposals_duplicate += 1
                continue
            seen_in_batch.add(dedup_key)
            session.add(
                CandidateEdge(
                    edge_type=proposal.edge_type,
                    src_version_id=src_id,
                    dst_version_id=dst_id,
                    src_title_raw=proposal.src_title,
                    dst_title_raw=proposal.dst_title,
                    supporting_sentence=proposal.supporting_sentence,
                    source_page=str(page["title"]),
                    source_revision=str(page["revision"]),
                    model_id=model_id,
                    model_confidence=proposal.confidence,
                    extraction_run=report.run_hash,
                )
            )
            report.candidates_written += 1
            if src_id is not None and dst_id is not None:
                report.resolved_both_ends += 1
            else:
                report.kept_raw_titles += 1
    session.flush()
    total_responses = report.pages_processed or len(raw_outputs)
    if total_responses:
        report.parse_failure_rate = round(report.responses_malformed / total_responses, 4)
    return report


def _bind_version(session: Session, title: str, language: str | None) -> Any:
    """Conservative title→version binding: unambiguous ≥0.9 hit or nothing (raw kept)."""
    result = resolve_title(session, title, language)
    if not result.candidates:
        return None
    top = result.candidates[0]
    if top.score < RESOLVE_BIND_THRESHOLD:
        return None
    contenders = [c for c in result.candidates if c.score >= RESOLVE_BIND_THRESHOLD]
    if len({c.work_id for c in contenders}) > 1:
        return None  # ambiguous at binding strength — the human reviewer decides
    return top.version_id


def collect_pages(session: Session, language: str = "en") -> dict[str, dict[str, Any]]:
    """The extraction input: gate-agnostic plot rows (content, not facts) per version."""
    rows = session.execute(
        select(PlotText, Version.title, Version.language, Version.release_year)
        .join(Version, Version.version_id == PlotText.version_id)
        .where(PlotText.source == "wikipedia", PlotText.language == language)
    ).all()
    pages: dict[str, dict[str, Any]] = {}
    for plot, version_title, version_language, release_year in rows:
        # Year disambiguates same-title works (Vikram 1986/2022, Devdas 1955/2002 — GS-10).
        key = f"{version_title} ({release_year})|{version_language}|{plot.language}"
        pages[key] = {
            "title": (plot.source_url or version_title).rsplit("/", 1)[-1],
            "text": plot.text,
            "revision": plot.revision_id or "",
        }
    return pages
