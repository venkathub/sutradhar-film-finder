"""Unit tests for the coverage/lift report computations (P1 task 13) — no DB."""

from __future__ import annotations

from pathlib import Path

import pytest

from sutradhar.pipeline.report import (
    FLAGSHIP_FRANCHISES,
    compute_edge_coverage,
    compute_version_coverage,
)
from sutradhar.pipeline.seed import DEFAULT_SEED_PATH, SeedSlice, load_seed_slice

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def slice_() -> SeedSlice:
    return load_seed_slice(REPO_ROOT / DEFAULT_SEED_PATH)


def _all_present(slice_: SeedSlice) -> set[tuple[str, str, int | None]]:
    return {
        (v.title, v.language, v.release_year)
        for w in slice_.works.values()
        for v in w.versions.values()
    }


def test_full_presence_is_full_coverage(slice_: SeedSlice) -> None:
    coverage = compute_version_coverage(slice_, _all_present(slice_))
    assert all(f.coverage == 1.0 and not f.missing for f in coverage)
    assert {f.franchise for f in coverage} >= set(FLAGSHIP_FRANCHISES)


def test_missing_version_lowers_coverage_and_is_named(slice_: SeedSlice) -> None:
    present = _all_present(slice_) - {("Papanasam", "ta", 2015)}
    coverage = compute_version_coverage(slice_, present)
    drishyam = next(f for f in coverage if f.franchise == "drishyam")
    assert drishyam.coverage < 1.0
    assert drishyam.missing == ["papanasam_ta"]
    # Other franchises unaffected (denominators are per-franchise, never blended).
    assert next(f for f in coverage if f.franchise == "baahubali").coverage == 1.0


def test_backlog_never_in_denominator(slice_: SeedSlice) -> None:
    """§7 Q1: conditional adds are name+reason rows — coverage cannot demand them."""
    total_expected = sum(f.expected for f in compute_version_coverage(slice_, set()))
    assert total_expected == slice_.version_count() == 31
    assert len(slice_.backlog) == 4  # exist, but in no denominator


def test_edge_coverage_counts_seed_relationships(slice_: SeedSlice) -> None:
    empty = compute_edge_coverage(slice_, set(), set())
    # 14 version relationships + 1 sequel + 3 based_on work links + 2 more…
    # exact denominator pinned so seed edits are conscious choices:
    assert empty.expected == 20
    assert empty.present == 0 and len(empty.missing) == 20


def test_edge_coverage_matches_proximate_targets(slice_: SeedSlice) -> None:
    """Chandramukhi counts only when the PROXIMATE edge (→ Apthamitra) is present."""
    direct_only = {("is_remake_of", "Chandramukhi|ta", "Manichitrathazhu|ml")}
    coverage = compute_edge_coverage(slice_, direct_only, set())
    assert any(m.startswith("chandramukhi_ta ") for m in coverage.missing)
    proximate = {("is_remake_of", "Chandramukhi|ta", "Apthamitra|kn")}
    coverage2 = compute_edge_coverage(slice_, proximate, set())
    assert not any(m.startswith("chandramukhi_ta ") for m in coverage2.missing)


def test_work_level_links_counted(slice_: SeedSlice) -> None:
    work_edges = {
        ("is_sequel_of", "drishyam_2", "drishyam"),
        ("based_on", "devadasu_1953", "devdas_novella"),
        ("based_on", "devdas_1955", "devdas_novella"),
        ("based_on", "devdas_2002", "devdas_novella"),
    }
    coverage = compute_edge_coverage(slice_, set(), work_edges)
    assert coverage.present == 4
    assert not any("devdas" in m or "drishyam_2 -is_sequel" in m for m in coverage.missing)
