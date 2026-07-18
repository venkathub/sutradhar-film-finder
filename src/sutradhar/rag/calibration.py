"""NO_MATCH θ binding — the threshold lives WITH the run that calibrated it.

P7 task 8 (DEC-P7-3; DEC-P7-1 finding 8). Previously ``retrieve.py`` hardcoded
``CALIBRATED_NO_MATCH_THRESHOLD`` as a numeric literal — silently decoupled from
the calibration artifact (run, config cell, embed model) it was derived from,
reusable against any future index without complaint. Now:

- θ is **read from the committed calibration artifact** of a single pinned run
  (``evals/retrieval_runs/<run>.calibration.json``, DEC-P2-5); no numeric θ
  literal exists in ``src/sutradhar/rag/`` (grep-tripwired in tests).
- Re-calibration = commit a new artifact + bump :data:`PINNED_CALIBRATION_RUN` —
  a reviewable one-line diff.
- :func:`assert_calibration_matches` is the **staleness gate** on the live
  serving path: if the index/embed-model/config-cell in use is not the one θ was
  calibrated on, serving HARD-FAILS with :class:`StaleCalibrationError` — never
  a silent reuse.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# DEC-P2-5 calibration run (P2 task 11). θ = 1.35 × top calibration-canary score
# (NEG-17); full curve + feasibility record in the artifact itself.
PINNED_CALIBRATION_RUN = "20260702T135315Z-f6583183"
DEFAULT_RUNS_DIR = Path("evals/retrieval_runs")


class StaleCalibrationError(RuntimeError):
    """The live retrieval stack does not match the run θ was calibrated on."""


@dataclass(frozen=True)
class CalibrationBinding:
    """θ plus the identity of everything it was calibrated against."""

    run_id: str
    theta: float
    config_key: str  # "<chunk_config>/d<rerank_depth>", e.g. "1024tok_15pct/d20"
    embed_model: str
    rerank_model: str

    @property
    def chunk_config(self) -> str:
        return self.config_key.split("/", 1)[0]

    @property
    def rerank_depth(self) -> int:
        return int(self.config_key.split("/d", 1)[1])


@lru_cache(maxsize=4)
def load_calibration(
    run_id: str = PINNED_CALIBRATION_RUN, runs_dir: Path = DEFAULT_RUNS_DIR
) -> CalibrationBinding:
    """Load the pinned run's θ + provenance from its committed artifacts."""
    calibration = json.loads((runs_dir / f"{run_id}.calibration.json").read_text(encoding="utf-8"))
    meta = json.loads((runs_dir / f"{run_id}.meta.json").read_text(encoding="utf-8"))["meta"]
    return CalibrationBinding(
        run_id=run_id,
        theta=float(calibration["theta"]),
        config_key=str(calibration["config_key"]),
        embed_model=str(meta["embed_model"]),
        rerank_model=str(meta["rerank_model"]),
    )


def calibrated_threshold() -> float:
    """The pinned θ — :class:`RetrievalConfig`'s default ``no_match_threshold``."""
    return load_calibration().theta


def assert_calibration_matches(
    *,
    embed_model: str,
    index_version: str,
    chunk_config: str,
    rerank_depth: int,
    binding: CalibrationBinding | None = None,
) -> CalibrationBinding:
    """Hard-fail unless the live stack matches the calibration run (DEC-P7-3).

    Called on the serving path before θ-based abstention goes live. Any mismatch
    means θ is being applied to scores it was never calibrated for — the exact
    silent-reuse failure P7 removes.
    """
    binding = binding or load_calibration()
    mismatches: list[str] = []
    if index_version != binding.run_id:
        mismatches.append(f"index_version {index_version!r} != calibration run {binding.run_id!r}")
    if embed_model != binding.embed_model:
        mismatches.append(f"embed_model {embed_model!r} != calibrated {binding.embed_model!r}")
    if chunk_config != binding.chunk_config:
        mismatches.append(f"chunk_config {chunk_config!r} != calibrated {binding.chunk_config!r}")
    if rerank_depth != binding.rerank_depth:
        mismatches.append(f"rerank_depth {rerank_depth} != calibrated {binding.rerank_depth}")
    if mismatches:
        raise StaleCalibrationError(
            "NO_MATCH θ is stale for the live retrieval stack — "
            + "; ".join(mismatches)
            + ". Recalibrate (evals/calibrate_no_match.py) against the current index "
            "and bump PINNED_CALIBRATION_RUN; θ is never silently reused (DEC-P7-3)."
        )
    return binding
