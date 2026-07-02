"""Hermetic tests for GPU-job input export (P2 task 5) — query collection reads repo
files only (golden YAML + negatives YAML), no DB. Corpus export is integration-tested."""

from __future__ import annotations

from sutradhar.rag.chunking import content_hash
from sutradhar.rag.gpu_jobs import collect_query_records, git_sha


def test_every_golden_and_negative_query_exported() -> None:
    records = collect_query_records()
    ids = {r["id"] for r in records}
    # All 11 golden categories contribute at least one query…
    for gs in [f"GS-{i:02d}" for i in range(1, 12)]:
        assert any(i.startswith(gs) for i in ids), f"{gs} missing"
    # …and all 24 held-out negatives ride along.
    assert sum(1 for i in ids if i.startswith("NEG-")) == 24


def test_multi_turn_fixtures_export_each_turn() -> None:
    records = collect_query_records()
    turn_ids = [r["id"] for r in records if "#turn" in r["id"]]
    assert turn_ids, "expected multi-turn (GS-08 backtracking) turns to be exported"
    assert all(i.split("#turn")[1].isdigit() for i in turn_ids)


def test_records_are_hash_keyed_sorted_unique() -> None:
    records = collect_query_records()
    assert [r["id"] for r in records] == sorted(r["id"] for r in records)
    assert len({r["id"] for r in records}) == len(records)
    for r in records:
        assert r["hash"] == content_hash(r["text"])  # the ArtifactEmbeddings lookup key


def test_git_sha_shape() -> None:
    sha = git_sha()
    assert sha is None or (len(sha) == 40 and all(c in "0123456789abcdef" for c in sha))
