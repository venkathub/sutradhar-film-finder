"""D3 structural exclusion rule — run BEFORE any training-slice ingestion (P4_SPEC §3 D3/§4).

Every candidate training-slice title (version titles, work primary titles, and backlog
names — franchise-level exclusion) must fall OUTSIDE the rapidfuzz-``MATCH_THRESHOLD``
(0.80) radius of every protected surface:

  (a) golden-slice titles (``data-pipeline/seed_slice.yaml`` — the entities the 12 frozen
      generation fixtures stand on),
  (b) GS-02 negative surfaces (``evals/golden/gs02_no_match.yaml`` queries + slot titles:
      Kaithi, Inception, Salaar, Pushpa, …),
  (c) held-out negative titles (``evals/negatives/heldout.yaml`` kind=title: Master,
      Pushpa: The Rise, Jailer, …) — their absence-at-radius is what θ (DEC-P2-5) was
      calibrated against,
  (d) the frozen exemplar-franchise titles (Ghajini, Okkadu, Ghilli, Interstellar —
      prompt surface for BOTH Table 2 columns; training on them would asymmetrically
      favour the QLoRA row).

NO_MATCH decoy *themes* (``data-pipeline/training_decoys.yaml``) must likewise clear the
negative plot-query radius, so out-of-catalog scaffolds can never teach an answer for a
query the negative suites assert must abstain.

Scorer: ``fuzz.ratio`` over ``match_key``-normalized strings — byte-identical to the
scoring backing ``resolve_title`` (DEC-P1-5), so "outside the radius" here means exactly
"invisible to the title channel" there. A violation means the training slice must be
re-authored; the threshold never moves.

Post-ingestion, ``tests/integration/test_negatives_absent.py`` re-asserts the negatives'
absence over the grown title index (same invariant, new population).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from sutradhar.pipeline.normalize import MATCH_THRESHOLD, best_matches, match_key
from sutradhar.pipeline.seed import DEFAULT_SEED_PATH, SeedSlice, load_seed_slice

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_SLICE_PATH = REPO_ROOT / DEFAULT_SEED_PATH
TRAINING_SLICE_PATH = REPO_ROOT / "data-pipeline" / "training_slice.yaml"
DECOYS_PATH = REPO_ROOT / "data-pipeline" / "training_decoys.yaml"
GS02_PATH = REPO_ROOT / "evals" / "golden" / "gs02_no_match.yaml"
HELDOUT_PATH = REPO_ROOT / "evals" / "negatives" / "heldout.yaml"
EXEMPLARS_PATH = REPO_ROOT / "evals" / "prompts" / "exemplars_v1.md"

# The frozen exemplar-franchise titles (DEC-P3-4 prompt surface; P4_SPEC §8.1).
# test_exemplar_titles_in_sync guards this list against the frozen artifact.
EXEMPLAR_TITLES = ("Ghajini", "Okkadu", "Ghilli", "Interstellar")


def _slice_titles(slice_: SeedSlice) -> set[str]:
    """Franchise-level title surface: versions + primary titles + backlog names."""
    titles: set[str] = set()
    for work in slice_.works.values():
        titles.add(work.primary_title)
        for version in work.versions.values():
            titles.add(version.title)
    titles.update(entry.name for entry in slice_.backlog)
    return titles


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _gs02_surfaces() -> tuple[set[str], set[str]]:
    """(title-like surfaces, plot-query surfaces) from GS-02 fixtures."""
    titles: set[str] = set()
    plots: set[str] = set()
    for fixture in _load_yaml(GS02_PATH)["fixtures"]:
        slots = fixture.get("expected_slots") or {}
        if "title" in slots:
            titles.add(slots["title"])
        # Queries are conservatively checked on BOTH surfaces: bare-title queries
        # (Kaithi, Inception) live in `query`; long plot queries can't collide with
        # short titles at 0.80 anyway, so over-inclusion is harmless.
        titles.add(fixture["query"])
        plots.add(fixture["query"])
    return titles, plots


def _heldout_surfaces() -> tuple[set[str], set[str]]:
    titles: set[str] = set()
    plots: set[str] = set()
    for fixture in _load_yaml(HELDOUT_PATH)["fixtures"]:
        if fixture["kind"] == "title":
            titles.add(fixture["query"])
        else:
            plots.add(fixture["query"])
    return titles, plots


def _protected_title_keys() -> dict[str, str]:
    """match_key -> human-readable provenance, over all protected title surfaces."""
    golden = _slice_titles(load_seed_slice(GOLDEN_SLICE_PATH))
    gs02_titles, _ = _gs02_surfaces()
    heldout_titles, _ = _heldout_surfaces()
    surfaces: dict[str, str] = {}
    for label, titles in (
        ("golden-slice", golden),
        ("GS-02", gs02_titles),
        ("held-out-negative", heldout_titles),
        ("exemplar", set(EXEMPLAR_TITLES)),
    ):
        for title in titles:
            surfaces.setdefault(match_key(title), f"{label}: {title!r}")
    return surfaces


@pytest.fixture(scope="module")
def training_slice() -> SeedSlice:
    if not TRAINING_SLICE_PATH.exists():
        pytest.fail(
            "data-pipeline/training_slice.yaml missing — the D3 exclusion rule must run "
            "against the slice config BEFORE any ingestion (P4_SPEC task 3)."
        )
    return load_seed_slice(TRAINING_SLICE_PATH)


def test_training_slice_loads_as_valid_seed_slice(training_slice: SeedSlice) -> None:
    """Same schema + structural validators as the golden slice (existing pipeline, D3)."""
    assert training_slice.works, "training slice must not be empty"
    assert len(training_slice.franchises()) >= 8  # ~10-15 target; hard floor


def test_training_franchises_disjoint_from_golden(training_slice: SeedSlice) -> None:
    golden = load_seed_slice(GOLDEN_SLICE_PATH)
    overlap = set(training_slice.franchises()) & set(golden.franchises())
    assert not overlap, f"training franchises collide with golden franchises: {sorted(overlap)}"
    key_overlap = set(training_slice.works) & set(golden.works)
    assert not key_overlap, f"training work keys collide with golden keys: {sorted(key_overlap)}"


def test_training_titles_outside_protected_radius(training_slice: SeedSlice) -> None:
    """THE exclusion rule: no training title within rapidfuzz-0.80 of any protected title."""
    protected = _protected_title_keys()
    protected_keys = sorted(protected)
    violations: list[str] = []
    for title in sorted(_slice_titles(training_slice)):
        for key, score in best_matches(
            match_key(title), protected_keys, limit=3, threshold=MATCH_THRESHOLD
        ):
            violations.append(f"training {title!r} ~ {protected[key]} (score {score})")
    assert not violations, "training-slice titles inside the protected 0.80 radius:\n" + "\n".join(
        violations
    )


def test_training_qids_disjoint_from_golden() -> None:
    """Entity-level disjointness at the source: no shared Wikidata QID across slices."""
    golden = load_seed_slice(GOLDEN_SLICE_PATH)
    training = load_seed_slice(TRAINING_SLICE_PATH)
    golden_qids = {qid for _, qid in golden._iter_qids()}
    training_qids = {qid for _, qid in training._iter_qids()}
    overlap = golden_qids & training_qids
    assert not overlap, f"QIDs present in both slices: {sorted(overlap)}"


def test_decoy_themes_clear_negative_radius() -> None:
    """NO_MATCH decoys are themes (not films) and stay off every negative plot query."""
    payload = _load_yaml(DECOYS_PATH)
    themes: list[str] = [d["theme"] for d in payload["decoy_themes"]]
    assert 2 <= len(themes) <= 5, "spec: 2-3 out-of-catalog decoy themes (small tolerance)"
    _, gs02_plots = _gs02_surfaces()
    _, heldout_plots = _heldout_surfaces()
    protected_plots = {match_key(q): q for q in gs02_plots | heldout_plots}
    protected_titles = _protected_title_keys()
    violations: list[str] = []
    for theme in themes:
        theme_key = match_key(theme)
        for key, score in best_matches(
            theme_key, sorted(protected_plots), limit=3, threshold=MATCH_THRESHOLD
        ):
            violations.append(f"decoy {theme!r} ~ negative plot {protected_plots[key]!r} ({score})")
        for key, score in best_matches(
            theme_key, sorted(protected_titles), limit=3, threshold=MATCH_THRESHOLD
        ):
            violations.append(f"decoy {theme!r} ~ {protected_titles[key]} ({score})")
    assert not violations, "decoy themes inside a protected radius:\n" + "\n".join(violations)


def test_exemplar_titles_in_sync() -> None:
    """The hardcoded exemplar list must match the frozen exemplars artifact."""
    text = EXEMPLARS_PATH.read_text(encoding="utf-8")
    for title in EXEMPLAR_TITLES:
        assert title in text, f"exemplar title {title!r} not found in {EXEMPLARS_PATH.name}"
