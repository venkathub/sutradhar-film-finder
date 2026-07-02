"""Hermetic unit tests for the retrieval artifact store (P2 task 4). tmp_path only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sutradhar.rag.artifacts import (
    ArtifactCorruptError,
    ArtifactEmbeddings,
    ArtifactRun,
    DenseSparse,
    MissingArtifactError,
    new_run_id,
    write_embedding_bank,
)
from sutradhar.rag.chunking import content_hash

TEXTS = ["Papanasam plot chunk.", "Drishyam plot chunk.", "வெறும் தமிழ் உரை."]


def _sealed_run(base: Path) -> ArtifactRun:
    run = ArtifactRun.create(base, run_id="run-test")
    dense = np.arange(9, dtype=np.float32).reshape(3, 3)
    sparse = [{1: 0.5}, {2: 0.25, 7: 1.0}, {250_001: 0.125}]
    write_embedding_bank(run, "queries", [content_hash(t) for t in TEXTS], dense, sparse)
    run.write_json("meta.json", {"run_id": "run-test", "banks": ["queries"]})
    run.write_manifest()
    return run


def test_run_id_shape_and_uniqueness() -> None:
    a, b = new_run_id(), new_run_id()
    assert a != b
    assert "T" in a and a.split("-")[0].endswith("Z")


def test_seal_open_verify_roundtrip(tmp_path: Path) -> None:
    _sealed_run(tmp_path)
    run = ArtifactRun.open(tmp_path, "run-test")  # verifies
    assert run.run_id == "run-test"
    manifest = run.path("MANIFEST.sha256").read_text(encoding="utf-8")
    assert "queries_dense.npy" in manifest and "meta.json" in manifest


def test_embeddings_lookup_exact_vectors(tmp_path: Path) -> None:
    run = _sealed_run(tmp_path)
    provider = ArtifactEmbeddings(run, banks=("queries",))
    assert len(provider) == 3
    rows = provider.embed([TEXTS[2], TEXTS[0]])  # any order, incl. native script
    assert isinstance(rows[0], DenseSparse)
    assert np.allclose(rows[0].dense, [6.0, 7.0, 8.0])
    assert rows[0].sparse == {250_001: 0.125}
    assert np.allclose(rows[1].dense, [0.0, 1.0, 2.0])
    assert rows[1].sparse == {1: 0.5}


def test_unseen_text_raises_missing_artifact(tmp_path: Path) -> None:
    provider = ArtifactEmbeddings(_sealed_run(tmp_path), banks=("queries",))
    with pytest.raises(MissingArtifactError, match="no recorded embedding"):
        provider.embed(["a query nobody embedded"])


def test_corrupt_file_is_a_hard_failure(tmp_path: Path) -> None:
    run = _sealed_run(tmp_path)
    path = run.path("queries_sparse.json")
    path.write_text(path.read_text(encoding="utf-8").replace("0.5", "0.9"), encoding="utf-8")
    with pytest.raises(ArtifactCorruptError, match="sha256 mismatch"):
        ArtifactRun.open(tmp_path, "run-test")
    with pytest.raises(ArtifactCorruptError):  # provider construction verifies too
        ArtifactEmbeddings(run, banks=("queries",))


def test_missing_and_stray_files_fail_verification(tmp_path: Path) -> None:
    run = _sealed_run(tmp_path)
    stray = run.path("stray.bin")
    stray.write_bytes(b"\x00")
    with pytest.raises(ArtifactCorruptError, match="unlisted"):
        run.verify()
    stray.unlink()
    run.path("meta.json").unlink()
    with pytest.raises(ArtifactCorruptError, match="missing"):
        run.verify()


def test_unsealed_run_cannot_be_opened(tmp_path: Path) -> None:
    ArtifactRun.create(tmp_path, run_id="unsealed").write_json("meta.json", {})
    with pytest.raises(ArtifactCorruptError, match="unsealed or missing"):
        ArtifactRun.open(tmp_path, "unsealed")


def test_empty_run_cannot_be_sealed(tmp_path: Path) -> None:
    run = ArtifactRun.create(tmp_path, run_id="empty")
    with pytest.raises(ArtifactCorruptError, match="empty"):
        run.write_manifest()


def test_misaligned_bank_rejected_at_write_and_read(tmp_path: Path) -> None:
    run = ArtifactRun.create(tmp_path, run_id="bad")
    dense = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="misaligned"):
        write_embedding_bank(run, "queries", ["h1"], dense, [{1: 1.0}, {2: 1.0}])
    # Doctor a row-count mismatch past the writer → reader still refuses.
    write_embedding_bank(run, "queries", ["h1", "h2"], dense, [{1: 1.0}, {2: 1.0}])
    run.path("queries_hashes.json").write_text(json.dumps(["h1"]), encoding="utf-8")
    run.write_manifest()
    with pytest.raises(ArtifactCorruptError, match="misaligned"):
        ArtifactEmbeddings(run, banks=("queries",))


def test_create_refuses_existing_run_dir(tmp_path: Path) -> None:
    ArtifactRun.create(tmp_path, run_id="dup")
    with pytest.raises(FileExistsError):
        ArtifactRun.create(tmp_path, run_id="dup")


def test_retrieval_run_settings_pin() -> None:
    """RETRIEVAL_RUN + EMBED_BASE_URL are env-driven, unset by default (P2_SPEC §2.7)."""
    from sutradhar.config import Settings

    assert Settings().retrieval_run is None or isinstance(Settings().retrieval_run, str)
    pinned = Settings(RETRIEVAL_RUN="20260702T120000Z-abcd1234", EMBED_BASE_URL="")
    assert pinned.retrieval_run == "20260702T120000Z-abcd1234"
    assert Settings(EMBED_BASE_URL="http://gpu:8001/v1").embed_base_url == "http://gpu:8001/v1"
