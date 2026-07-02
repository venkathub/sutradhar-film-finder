"""Versioned retrieval artifacts (P2 task 4): run dirs, MANIFEST.sha256, load API.

Reuses the P1 snapshot discipline (``data/raw/<src>/<ts>/ + MANIFEST.sha256``) for the
GPU session's outputs: every embedding/score file under
``data/artifacts/retrieval/<run_id>/`` is sha256-recorded, and **nothing is served from a
run that fails verification** — a corrupt or tampered artifact is a hard failure, never a
silent fallback (P2_SPEC §4). Raw runs are git-ignored; the compact eval summary that CI
consumes is committed separately under ``evals/retrieval_runs/`` (DEC-P2-6).

Embedding banks (the ``ArtifactEmbeddings`` storage convention, written by the task-5 GPU
job and read on the laptop/CI):

- ``<bank>_hashes.json`` — row-ordered list of ``sha256(text)`` keys;
- ``<bank>_dense.npy``   — float32 ``[N, dim]`` BGE-M3 dense vectors;
- ``<bank>_sparse.json`` — row-aligned list of ``{token_id: weight}`` lexical weights.

``ArtifactEmbeddings`` implements the ``EmbeddingProvider`` protocol by *lookup*, keyed by
``sha256(text)``: recorded texts embed instantly with zero GPU; an unseen text raises
:class:`MissingArtifactError` — the laptop/CI path never degrades to a fake vector.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from sutradhar.rag.chunking import content_hash

DEFAULT_ARTIFACTS_ROOT = Path("data/artifacts/retrieval")
MANIFEST_NAME = "MANIFEST.sha256"


class ArtifactCorruptError(RuntimeError):
    """A run failed MANIFEST verification (missing/mismatched/unlisted file)."""


class MissingArtifactError(KeyError):
    """A text has no recorded embedding in this run (never silently degrade)."""


@dataclass(frozen=True)
class DenseSparse:
    """One BGE-M3 embedding: dense vector + sparse lexical weights (token id → weight)."""

    dense: npt.NDArray[np.float32]
    sparse: dict[int, float]


class EmbeddingProvider(Protocol):  # P2_SPEC §2.5
    def embed(self, texts: list[str]) -> list[DenseSparse]: ...


def new_run_id(now: datetime | None = None) -> str:
    """Timestamped, collision-safe run id, e.g. ``20260702T142501Z-3f9a1c2b``."""
    stamp = (now or datetime.now(tz=UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(4)}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class ArtifactRun:
    """One versioned artifact run directory. Write → seal (manifest) → open (verified)."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def run_id(self) -> str:
        return self.root.name

    @classmethod
    def create(
        cls, base_dir: Path = DEFAULT_ARTIFACTS_ROOT, run_id: str | None = None
    ) -> ArtifactRun:
        run = cls(base_dir / (run_id or new_run_id()))
        run.root.mkdir(parents=True, exist_ok=False)
        return run

    @classmethod
    def open(cls, base_dir: Path, run_id: str) -> ArtifactRun:
        """Open an existing run, verifying its MANIFEST before anything is served."""
        run = cls(base_dir / run_id)
        run.verify()
        return run

    def path(self, name: str) -> Path:
        return self.root / name

    # --- Writing (GPU job side) ---

    def write_json(self, name: str, payload: object) -> Path:
        path = self.path(name)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8"
        )
        return path

    def write_dense(self, name: str, array: npt.NDArray[np.float32]) -> Path:
        path = self.path(name)
        np.save(path, np.ascontiguousarray(array, dtype=np.float32))
        return path

    def write_manifest(self) -> Path:
        """Seal the run: hash every file (sorted, manifest excluded). Empty run = error."""
        files = sorted(p for p in self.root.rglob("*") if p.is_file() and p.name != MANIFEST_NAME)
        if not files:
            raise ArtifactCorruptError(f"refusing to seal empty artifact run {self.root}")
        lines = [f"{_sha256_file(p)}  {p.relative_to(self.root).as_posix()}" for p in files]
        manifest = self.path(MANIFEST_NAME)
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return manifest

    # --- Verification (laptop/CI side) ---

    def verify(self) -> None:
        """Hard-fail on: no manifest, missing file, hash mismatch, or unlisted stray file."""
        manifest = self.path(MANIFEST_NAME)
        if not manifest.is_file():
            raise ArtifactCorruptError(
                f"no {MANIFEST_NAME} in {self.root} — unsealed or missing run"
            )
        recorded: dict[str, str] = {}
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, _, name = line.partition("  ")
                recorded[name] = digest
        present = {
            p.relative_to(self.root).as_posix()
            for p in self.root.rglob("*")
            if p.is_file() and p.name != MANIFEST_NAME
        }
        if missing := sorted(set(recorded) - present):
            raise ArtifactCorruptError(f"artifact run {self.run_id}: missing files {missing}")
        if strays := sorted(present - set(recorded)):
            raise ArtifactCorruptError(f"artifact run {self.run_id}: unlisted files {strays}")
        for name, digest in recorded.items():
            if _sha256_file(self.root / name) != digest:
                raise ArtifactCorruptError(f"artifact run {self.run_id}: sha256 mismatch on {name}")


# --- Embedding banks ---


def write_embedding_bank(
    run: ArtifactRun,
    bank: str,
    hashes: list[str],
    dense: npt.NDArray[np.float32],
    sparse: list[dict[int, float]],
) -> None:
    """Persist one row-aligned bank (see module docstring for the file convention)."""
    if not (len(hashes) == dense.shape[0] == len(sparse)):
        raise ValueError(
            f"bank {bank!r}: misaligned rows "
            f"(hashes={len(hashes)}, dense={dense.shape[0]}, sparse={len(sparse)})"
        )
    run.write_json(f"{bank}_hashes.json", hashes)
    run.write_dense(f"{bank}_dense.npy", dense)
    run.write_json(f"{bank}_sparse.json", [{str(k): v for k, v in row.items()} for row in sparse])


class ArtifactEmbeddings:
    """Recorded-vector ``EmbeddingProvider``: lookup by ``sha256(text)``, verified-run only."""

    def __init__(self, run: ArtifactRun, banks: tuple[str, ...]) -> None:
        run.verify()  # never serve a single vector from an unverified run
        self._run_id = run.run_id
        self._rows: dict[str, DenseSparse] = {}
        for bank in banks:
            hashes: list[str] = json.loads(
                run.path(f"{bank}_hashes.json").read_text(encoding="utf-8")
            )
            dense = np.load(run.path(f"{bank}_dense.npy"))
            sparse_raw: list[dict[str, float]] = json.loads(
                run.path(f"{bank}_sparse.json").read_text(encoding="utf-8")
            )
            if not (len(hashes) == dense.shape[0] == len(sparse_raw)):
                raise ArtifactCorruptError(f"run {run.run_id}: bank {bank!r} rows misaligned")
            for key, vector, weights in zip(hashes, dense, sparse_raw, strict=True):
                self._rows[key] = DenseSparse(
                    dense=vector.astype(np.float32),
                    sparse={int(k): float(v) for k, v in weights.items()},
                )

    def __len__(self) -> int:
        return len(self._rows)

    def embed(self, texts: list[str]) -> list[DenseSparse]:
        out: list[DenseSparse] = []
        for text in texts:
            key = content_hash(text)
            row = self._rows.get(key)
            if row is None:
                raise MissingArtifactError(
                    f"run {self._run_id}: no recorded embedding for text "
                    f"{text[:60]!r}… (sha256 {key[:12]}…) — re-run the GPU embed job"
                )
            out.append(row)
        return out
