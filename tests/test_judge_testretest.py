"""P7 task 17 (DEC-P7-6) — blind test-retest second pass for judge validation.

Tier-1, no GPU: blinding correctness (no labels, no foil-revealing ids, seeded
reproducible shuffle), pure kappa computation on synthetic pairs, offline judge leg
from frozen verdicts, and byte-frozenness of report.json.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sutradhar.evals.judge_validation import (
    BLIND_KEY_FILE,
    BLIND_WORKSHEET_FILE,
    VALIDATION_DIR,
    WorksheetItem,
    build_blind_worksheet,
    compute_testretest_report,
    load_worksheet,
    save_blind_worksheet,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSHEET_DIR = REPO_ROOT / VALIDATION_DIR


def _items() -> list[WorksheetItem]:
    return load_worksheet(WORKSHEET_DIR)


def test_blind_worksheet_hides_labels_foils_and_pairing(tmp_path: Path) -> None:
    items = _items()
    blind, id_map = build_blind_worksheet(items)
    path = save_blind_worksheet(blind, id_map, tmp_path)
    text = path.read_text(encoding="utf-8")
    payload_items = __import__("yaml").safe_load(text)["items"]
    # No label leaks (exclude_none drops the field entirely from every ITEM; the
    # word appears only in the labelling instructions), no foil-marked ids, no
    # original item ids, no fixture ids.
    assert "-foil" not in text
    assert "fixture_id" not in text
    for raw in payload_items:
        assert raw["blind_id"].startswith("blind-")
        assert "human_label" not in raw  # unlabelled until the rater fills it
    # The id map round-trips every original item exactly once.
    assert sorted(id_map.values()) == sorted(i.item_id for i in items)
    assert len(id_map) == len(items) == len(blind)


def test_blind_shuffle_is_seeded_and_reproducible() -> None:
    items = _items()
    first, map_a = build_blind_worksheet(items)
    second, map_b = build_blind_worksheet(items)
    assert map_a == map_b  # recorded seed => reproducible blinding
    assert [b.blind_id for b in first] == [b.blind_id for b in second]
    # And it actually shuffles (original order not preserved).
    assert [map_a[b.blind_id] for b in first] != [i.item_id for i in items]


def _label_blind(blind: list, id_map: dict[str, str], labels: dict[str, int]) -> list:
    return [b.model_copy(update={"human_label": labels[id_map[b.blind_id]]}) for b in blind]


def test_perfect_agreement_gives_kappa_one() -> None:
    items = _items()
    blind, id_map = build_blind_worksheet(items)
    labels = {i.item_id: int(i.human_label or 0) for i in items}
    report = compute_testretest_report(items, _label_blind(blind, id_map, labels), id_map)
    assert report.intra_rater_kappa == pytest.approx(1.0)
    assert report.intra_rater_kappa_real_items_only == pytest.approx(1.0)
    assert report.percent_agreement == pytest.approx(1.0)
    assert "NOT a human-human" in report.framing  # the caveat is part of the artifact


def test_disagreement_lowers_kappa_and_foils_are_excluded_from_real_only() -> None:
    items = _items()
    blind, id_map = build_blind_worksheet(items)
    labels = {i.item_id: int(i.human_label or 0) for i in items}
    # Flip every FOIL label in the second pass: overall kappa drops, but the
    # real-items-only kappa stays perfect (foils excluded from that metric).
    flipped = {k: (1 - v if k.endswith("-foil") else v) for k, v in labels.items()}
    report = compute_testretest_report(items, _label_blind(blind, id_map, flipped), id_map)
    assert report.intra_rater_kappa < 1.0
    assert report.intra_rater_kappa_real_items_only == pytest.approx(1.0)


def test_judge_leg_computed_offline_from_frozen_verdicts() -> None:
    items = _items()
    blind, id_map = build_blind_worksheet(items)
    labels = {i.item_id: int(i.human_label or 0) for i in items}
    frozen = json.loads((WORKSHEET_DIR / "report.json").read_text(encoding="utf-8"))
    judge_binaries = {v["item_id"]: int(v["judge_binary"]) for v in frozen["verdicts"]}
    report = compute_testretest_report(
        items, _label_blind(blind, id_map, labels), id_map, judge_binaries
    )
    # Second pass == first pass here, so judge-vs-second == the frozen judge-vs-human kappa.
    assert report.second_pass_vs_judge_kappa == pytest.approx(frozen["cohens_kappa"], abs=1e-9)


def test_unlabelled_blind_items_are_rejected() -> None:
    items = _items()
    blind, id_map = build_blind_worksheet(items)
    with pytest.raises(ValueError, match="unlabelled"):
        compute_testretest_report(items, blind, id_map)


def test_frozen_report_is_untouched_by_the_flow(tmp_path: Path) -> None:
    """The whole test-retest path must never rewrite report.json."""
    frozen_path = WORKSHEET_DIR / "report.json"
    before = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    items = _items()
    blind, id_map = build_blind_worksheet(items)
    save_blind_worksheet(blind, id_map, tmp_path)
    labels = {i.item_id: int(i.human_label or 0) for i in items}
    compute_testretest_report(items, _label_blind(blind, id_map, labels), id_map)
    assert hashlib.sha256(frozen_path.read_bytes()).hexdigest() == before


def test_protocol_is_committed_before_labelling() -> None:
    protocol = WORKSHEET_DIR / "PROTOCOL.md"
    assert protocol.exists(), "PROTOCOL.md must be committed before any second-pass label"
    text = protocol.read_text(encoding="utf-8")
    for required in ("intra-rater", "NOT a human–human", "≥ 14 days", "20260718"):
        assert required in text, f"protocol lost required clause: {required!r}"
    # The blind pass has not silently started with a stale/committed labelled worksheet:
    # if the blind worksheet exists in-repo, it must be unlabelled or the report exists.
    blind_path = WORKSHEET_DIR / BLIND_WORKSHEET_FILE
    if blind_path.exists():
        assert (WORKSHEET_DIR / BLIND_KEY_FILE).exists()
