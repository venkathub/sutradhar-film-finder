"""Scaffold source snapshot: gate-view recordings the generator runs from (P4 task 4).

The scaffold generator (``sutradhar.finetune.scaffold``) is a PURE function of
``(snapshot, seed, config)`` so Tier-1 CI can run it without a database. The snapshot is
exported once per training-slice ingestion by ``finetune/build_dataset.py snapshot``,
which calls the five v0 repository functions against the **gate-visible views only** and
records their ``model_dump`` outputs verbatim — training tool results are therefore
byte-what the live tools return (CANDIDATE-tier rows can never appear, the same layered-
gate property as P1).

``sha256`` of the committed file is the ``DatasetCard.graph_snapshot`` stamp.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

SNAPSHOT_PATH = Path("finetune/scaffold_snapshot.json")


class PlotExcerpt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str
    excerpt: str


class WorkSnapshot(BaseModel):
    """Recorded v0 tool results + prose raw material for one training-slice work."""

    model_config = ConfigDict(extra="forbid")

    work_key: str  # slice key (training_slice.yaml)
    franchise: str
    work_id: str
    canonical_title: str
    original_language: str | None
    get_work: dict[str, Any]  # recorded v0 get_work result
    get_versions: dict[str, dict[str, Any]]  # variant ("indian" | "indian_sequels") -> result
    resolve_title: dict[str, dict[str, Any]]  # query string -> recorded v0 result
    plot_excerpts: list[PlotExcerpt]


class DecoyTheme(BaseModel):
    model_config = ConfigDict(extra="forbid")

    theme: str
    query_lang: str


class ScaffoldSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slice_config: str
    tool_schema_sha256: str
    works: list[WorkSnapshot]
    decoy_themes: list[DecoyTheme]

    def work_by_key(self) -> dict[str, WorkSnapshot]:
        return {w.work_key: w for w in self.works}


def load_scaffold_snapshot(path: Path = SNAPSHOT_PATH) -> ScaffoldSnapshot:
    return ScaffoldSnapshot.model_validate_json(path.read_text(encoding="utf-8"))


def write_scaffold_snapshot(path: Path, snapshot: ScaffoldSnapshot) -> str:
    """Deterministic write (sorted keys, 2-space indent); returns the file sha256."""
    payload = (
        json.dumps(snapshot.model_dump(mode="json"), sort_keys=True, indent=2, ensure_ascii=False)
        + "\n"
    )
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def snapshot_sha256(path: Path = SNAPSHOT_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- Deterministic title perturbations (GS-11-style; shared by exporter + generator) ---

_DOUBLED_CONSONANT_RE = re.compile(r"([b-df-hj-np-tv-z])\1", re.IGNORECASE)


def title_perturbations(title: str) -> list[str]:
    """Misspelled/romanization-variant probes for ``resolve_title`` recordings.

    Pure + deterministic: the exporter records ``resolve_title`` for each candidate and
    keeps only those that still resolve (≥1 candidate) — the generator then samples only
    recorded, resolvable queries. Ops mirror common romanization drift: doubled-consonant
    collapse (Pokkiri→Pokiri), aa-flattening (Kaavalan→Kavalan), i→ee stretch
    (Bigil→Beegil), space-collapse (U Turn→UTurn).
    """
    candidates = []
    collapsed = _DOUBLED_CONSONANT_RE.sub(r"\1", title)
    candidates.append(collapsed)
    candidates.append(title.replace("aa", "a"))
    if "i" in title:
        candidates.append(title.replace("i", "ee", 1))
    if " " in title:
        candidates.append(title.replace(" ", ""))
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        if c and c != title and c.casefold() not in seen:
            seen.add(c.casefold())
            out.append(c)
    return out


# --- v0 result subschema (mirror of driver.params_subschema, result side) ---


def result_subschema(schema: dict[str, Any], tool: str) -> dict[str, Any]:
    """The tool's result schema with the root $defs attached — used to validate every
    recorded/constructed ``tool_result`` in training data against frozen v0."""
    sub = dict(schema["tools"][tool]["result"])
    sub["$defs"] = schema["$defs"]
    return sub
