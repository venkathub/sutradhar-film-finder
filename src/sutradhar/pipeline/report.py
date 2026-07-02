"""Graph coverage + extraction-lift report (P1 task 13, §1.9/§1.10).

Two metric families, kept apart from P2's retrieval metrics (different denominators):

- **Graph coverage** (the P1 exit gate): per franchise,
  ``gate_visible_versions / versions_in_curated_truth`` — the curated truth being the
  committed seed slice (backlog rows excluded by construction). Exit requires **1.0 on the
  flagship franchises**. Supplementary: curated-relationship edge coverage (which seed
  relationships have a gate-visible edge of the right type).
- **Extraction lift** (report, not gate): candidates proposed/confirmed/rejected, precision
  = confirmed/decided, parse-failure rate from the recorded artifact, and **verified edges
  added beyond Wikidata** (edges whose provenance has no wikidata ref) vs corroborations.

Every report carries the §6.1 reproducibility stamp: code SHA, seed-slice sha256, snapshot
manifest digests, extraction model + run hash.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from sutradhar.graph.schema import CandidateEdge
from sutradhar.pipeline.seed import DEFAULT_SEED_PATH, SeedSlice, load_seed_slice

FLAGSHIP_FRANCHISES = ("drishyam", "baahubali", "devdas", "vikram", "manichitrathazhu")


@dataclass
class FranchiseCoverage:
    franchise: str
    expected: int
    present: int
    missing: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        return round(self.present / self.expected, 4) if self.expected else 1.0


@dataclass
class EdgeCoverage:
    expected: int = 0
    present: int = 0
    missing: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        return round(self.present / self.expected, 4) if self.expected else 1.0


@dataclass
class ExtractionMetrics:
    proposed: int = 0
    confirmed: int = 0
    rejected: int = 0
    pending: int = 0  # skipped / undecided — excluded from the precision denominator
    edges_created_beyond_wikidata: int = 0
    edges_corroborated: int = 0
    pages: int = 0
    malformed: int = 0
    model_id: str = ""
    run_hash: str = ""

    @property
    def decided(self) -> int:
        return self.confirmed + self.rejected

    @property
    def precision(self) -> float | None:
        return round(self.confirmed / self.decided, 4) if self.decided else None

    @property
    def parse_failure_rate(self) -> float | None:
        return round(self.malformed / self.pages, 4) if self.pages else None


@dataclass
class GraphReport:
    franchises: list[FranchiseCoverage]
    edge_coverage: EdgeCoverage
    extraction: ExtractionMetrics
    stamp: dict[str, str]

    @property
    def flagship_coverage_ok(self) -> bool:
        return all(f.coverage == 1.0 for f in self.franchises if f.franchise in FLAGSHIP_FRANCHISES)


def compute_version_coverage(
    slice_: SeedSlice, present: set[tuple[str, str, int | None]]
) -> list[FranchiseCoverage]:
    """Coverage per franchise. ``present`` = gate-visible (title, language, year) triples."""
    results = []
    for franchise, work_keys in sorted(slice_.franchises().items()):
        expected = 0
        found = 0
        missing: list[str] = []
        for wkey in work_keys:
            for vkey, version in slice_.works[wkey].versions.items():
                expected += 1
                if (version.title, version.language, version.release_year) in present:
                    found += 1
                else:
                    missing.append(vkey)
        results.append(
            FranchiseCoverage(
                franchise=franchise, expected=expected, present=found, missing=missing
            )
        )
    return results


def compute_edge_coverage(
    slice_: SeedSlice,
    present_edges: set[tuple[str, str, str]],
    work_edges: set[tuple[str, str, str]],
) -> EdgeCoverage:
    """Curated-relationship coverage: seed version relationships + work-level lineage.

    ``present_edges``: gate-visible (edge_type, src_version_key-ish, dst_version_key-ish)
    resolved by (title, language) matching; ``work_edges``: (edge_type, src_work, dst_work).
    """
    result = EdgeCoverage()
    for wkey, work in slice_.works.items():
        for vkey, version in work.versions.items():
            if version.relationship is None:
                continue
            result.expected += 1
            target = work.versions[version.relationship.of]
            key = (
                version.relationship.type,
                f"{version.title}|{version.language}",
                f"{target.title}|{target.language}",
            )
            if key in present_edges:
                result.present += 1
            else:
                result.missing.append(
                    f"{vkey} -{version.relationship.type}-> {version.relationship.of}"
                )
        if work.is_sequel_of is not None:
            result.expected += 1
            if ("is_sequel_of", wkey, work.is_sequel_of) in work_edges:
                result.present += 1
            else:
                result.missing.append(f"{wkey} -is_sequel_of-> {work.is_sequel_of}")
        if work.based_on is not None:
            result.expected += 1
            if ("based_on", wkey, work.based_on) in work_edges:
                result.present += 1
            else:
                result.missing.append(f"{wkey} -based_on-> {work.based_on}")
    return result


def _gate_visible_versions(session: Session) -> set[tuple[str, str, int | None]]:
    rows = session.execute(
        text("SELECT title, language, release_year FROM ground_truth_versions")
    ).all()
    return {(r.title, r.language, r.release_year) for r in rows}


def _gate_visible_edges(
    session: Session, slice_: SeedSlice
) -> tuple[set[tuple[str, str, str]], set[tuple[str, str, str]]]:
    """(version-level edge keys, work-level edge keys) from the ground-truth views."""
    version_label: dict[object, str] = {
        row.version_id: row.label
        for row in session.execute(
            text(
                "SELECT v.version_id, v.title || '|' || v.language AS label "
                "FROM ground_truth_versions v"
            )
        ).all()
    }
    # Map DB works back to seed keys via (primary_title, first_release_year).
    seed_work_key = {
        (w.primary_title, w.first_release_year): key for key, w in slice_.works.items()
    }
    work_key: dict[object, str | None] = {}
    for row in session.execute(
        text("SELECT work_id, primary_title, first_release_year FROM ground_truth_works")
    ).all():
        work_key[row.work_id] = seed_work_key.get((row.primary_title, row.first_release_year))
    edge_rows = session.execute(
        text("SELECT edge_type, src_kind, src_id, dst_kind, dst_id FROM ground_truth_edges")
    ).all()
    version_edges: set[tuple[str, str, str]] = set()
    work_edges: set[tuple[str, str, str]] = set()
    for r in edge_rows:
        if r.src_kind == "version":
            src, dst = version_label.get(r.src_id), version_label.get(r.dst_id)
            if src and dst:
                version_edges.add((r.edge_type, src, dst))
        else:
            src_k, dst_k = work_key.get(r.src_id), work_key.get(r.dst_id)
            if src_k and dst_k:
                work_edges.add((r.edge_type, src_k, dst_k))
    return version_edges, work_edges


def compute_extraction_metrics(
    session: Session, artifact_dir: Path | None = None
) -> ExtractionMetrics:
    metrics = ExtractionMetrics()
    for candidate in session.scalars(select(CandidateEdge)).all():
        metrics.proposed += 1
        if candidate.status == "confirmed":
            metrics.confirmed += 1
        elif candidate.status == "rejected":
            metrics.rejected += 1
        else:
            metrics.pending += 1
        if candidate.extraction_run and not metrics.run_hash:
            metrics.run_hash = candidate.extraction_run
        if candidate.model_id and not metrics.model_id:
            metrics.model_id = candidate.model_id

    # Lift attribution via provenance: created-by-review = wikipedia+human, no wikidata.
    rows = session.execute(text("SELECT sources FROM ground_truth_edges")).all()
    for (sources,) in rows:
        source_ids = {s.get("source") for s in sources}
        if "wikipedia" in source_ids and "wikidata" not in source_ids:
            metrics.edges_created_beyond_wikidata += 1
        elif "wikipedia" in source_ids and "wikidata" in source_ids:
            metrics.edges_corroborated += 1

    if artifact_dir is not None and (artifact_dir / "outputs.json").exists():
        from sutradhar.pipeline.extract import parse_extraction_output
        from sutradhar.pipeline.snapshots import load_snapshot

        artifact = load_snapshot(artifact_dir, "outputs")
        raw_outputs = artifact.get("raw_outputs", {})
        metrics.pages = len(raw_outputs)
        metrics.malformed = sum(
            1 for raw in raw_outputs.values() if parse_extraction_output(str(raw)) is None
        )
        if not metrics.model_id:
            metrics.model_id = str(artifact.get("model_id", ""))
    return metrics


def _snapshot_digest(root: Path) -> str | None:
    """Digest of the latest snapshot's manifest under ``root`` (or None)."""
    if not root.exists():
        return None
    dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not dirs:
        return None
    manifest = dirs[-1] / "MANIFEST.sha256"
    if not manifest.exists():
        return None
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()[:12]
    return f"{dirs[-1].name}:{digest}"


def build_stamp(seed_path: Path, data_root: Path = Path("data/raw")) -> dict[str, str]:
    stamp: dict[str, str] = {}
    try:
        stamp["code_sha"] = subprocess.run(  # noqa: S603, S607 — git describe of own repo
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        stamp["code_sha"] = "unknown"
    stamp["seed_slice_sha"] = hashlib.sha256(seed_path.read_bytes()).hexdigest()[:12]
    for name in ("wikidata", "tmdb", "imdb", "wikipedia", "extraction"):
        digest = _snapshot_digest(data_root / name)
        if digest:
            stamp[f"snapshot_{name}"] = digest
    return stamp


def build_report(
    session: Session,
    seed_path: Path = DEFAULT_SEED_PATH,
    data_root: Path = Path("data/raw"),
) -> GraphReport:
    slice_ = load_seed_slice(seed_path)
    franchises = compute_version_coverage(slice_, _gate_visible_versions(session))
    version_edges, work_edges = _gate_visible_edges(session, slice_)
    edge_coverage = compute_edge_coverage(slice_, version_edges, work_edges)
    extraction_dirs = (
        sorted((data_root / "extraction").glob("*")) if (data_root / "extraction").exists() else []
    )
    extraction = compute_extraction_metrics(
        session, extraction_dirs[-1] if extraction_dirs else None
    )
    return GraphReport(
        franchises=franchises,
        edge_coverage=edge_coverage,
        extraction=extraction,
        stamp=build_stamp(seed_path, data_root),
    )


def render_report(report: GraphReport) -> str:
    lines = ["# Graph coverage & extraction lift", ""]
    lines.append("## Version coverage (gate-visible vs curated truth)")
    for f in report.franchises:
        flag = " [FLAGSHIP]" if f.franchise in FLAGSHIP_FRANCHISES else ""
        lines.append(
            f"  {f.franchise:18} {f.present}/{f.expected}  coverage={f.coverage:.2f}{flag}"
            + (f"  MISSING: {', '.join(f.missing)}" if f.missing else "")
        )
    lines.append(f"  flagship gate (=1.0): {'PASS' if report.flagship_coverage_ok else 'FAIL'}")
    ec = report.edge_coverage
    lines.append("")
    lines.append("## Curated-relationship edge coverage (supplementary)")
    lines.append(f"  {ec.present}/{ec.expected}  coverage={ec.coverage:.2f}")
    for m in ec.missing:
        lines.append(f"    missing: {m}")
    ex = report.extraction
    lines.append("")
    lines.append("## Extraction lift (report, not gate)")
    lines.append(f"  model: {ex.model_id}  run: {ex.run_hash}")
    lines.append(
        f"  pages: {ex.pages}  parse-failure rate: "
        f"{ex.parse_failure_rate if ex.parse_failure_rate is not None else 'n/a'}"
    )
    lines.append(
        f"  candidates: {ex.proposed} proposed → {ex.confirmed} confirmed / "
        f"{ex.rejected} rejected / {ex.pending} pending"
    )
    lines.append(f"  candidate precision (confirmed/decided): {ex.precision}")
    lines.append(
        f"  verified edges beyond Wikidata: {ex.edges_created_beyond_wikidata}  "
        f"(corroborated existing: {ex.edges_corroborated})"
    )
    lines.append("")
    lines.append("## Reproducibility stamp")
    for key, value in report.stamp.items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def report_to_json(report: GraphReport) -> str:
    return json.dumps(
        {
            "franchises": [
                {
                    "franchise": f.franchise,
                    "present": f.present,
                    "expected": f.expected,
                    "coverage": f.coverage,
                    "missing": f.missing,
                }
                for f in report.franchises
            ],
            "flagship_coverage_ok": report.flagship_coverage_ok,
            "edge_coverage": {
                "present": report.edge_coverage.present,
                "expected": report.edge_coverage.expected,
                "coverage": report.edge_coverage.coverage,
                "missing": report.edge_coverage.missing,
            },
            "extraction": {
                "proposed": report.extraction.proposed,
                "confirmed": report.extraction.confirmed,
                "rejected": report.extraction.rejected,
                "pending": report.extraction.pending,
                "precision": report.extraction.precision,
                "parse_failure_rate": report.extraction.parse_failure_rate,
                "edges_created_beyond_wikidata": report.extraction.edges_created_beyond_wikidata,
                "edges_corroborated": report.extraction.edges_corroborated,
                "model_id": report.extraction.model_id,
                "run_hash": report.extraction.run_hash,
            },
            "stamp": report.stamp,
        },
        indent=1,
    )
