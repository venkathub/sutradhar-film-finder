"""Integration: scaffold-snapshot export is deterministic and gate-view-faithful (P4 task 4).

- Exporting twice against the same DB produces byte-identical files (the hash the card's
  ``graph_snapshot`` pins is reproducible).
- ``refine_local`` (the generator's client-side refine over recorded rows) agrees with the
  live ``repository.refine_filter`` for every dimension the scaffolds use — so constructed
  refine results are byte-what the live tool would return.
- Recorded rows come from gate views only: every recorded version id is gate-visible.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session

from sutradhar.finetune.scaffold import refine_local
from sutradhar.finetune.snapshot import load_scaffold_snapshot
from sutradhar.graph import repository
from sutradhar.graph.db import postgres_url

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SNAPSHOT = REPO_ROOT / "finetune" / "scaffold_snapshot.json"


@pytest.fixture(scope="module")
def engine() -> Iterator[Engine]:
    eng = create_engine(postgres_url())
    try:
        with eng.connect():
            pass
    except Exception as exc:  # noqa: BLE001 — connection probe
        pytest.skip(f"Postgres not reachable ({exc}); run `make up` first.")
    yield eng
    eng.dispose()


@pytest.fixture()
def session(engine: Engine) -> Iterator[Session]:
    with Session(engine) as s:
        yield s


def _snapshot_matches_db(session: Session) -> bool:
    """The committed snapshot only replays against the DB build it was exported from."""
    snap = load_scaffold_snapshot(SNAPSHOT)
    probe = snap.works[0]
    row = session.execute(
        text("SELECT count(*) FROM ground_truth_works WHERE work_id = :wid"),
        {"wid": probe.work_id},
    ).scalar_one()
    return bool(row)


def test_snapshot_export_is_deterministic(session: Session, tmp_path: Path) -> None:
    if not _snapshot_matches_db(session):
        pytest.skip("DB was rebuilt since the committed snapshot; re-export to compare.")
    import subprocess
    import sys

    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    for out in (out_a, out_b):
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "finetune" / "build_dataset.py"),
                "snapshot",
                "--out",
                str(out),
            ],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
        )
    assert out_a.read_bytes() == out_b.read_bytes()
    assert out_a.read_bytes() == SNAPSHOT.read_bytes(), (
        "committed scaffold_snapshot.json is stale vs the live gate views — re-export it"
    )


def test_recorded_versions_are_gate_visible(session: Session) -> None:
    if not _snapshot_matches_db(session):
        pytest.skip("DB was rebuilt since the committed snapshot; re-export to compare.")
    snap = load_scaffold_snapshot(SNAPSHOT)
    visible = {
        str(r.version_id)
        for r in session.execute(text("SELECT version_id FROM ground_truth_versions")).all()
    }
    for work in snap.works:
        for variant in work.get_versions.values():
            for entry in variant.get("versions", []):
                assert str(entry["version_id"]) in visible, (
                    f"{work.work_key}: recorded version not gate-visible"
                )


def test_refine_local_matches_live_refine_filter(session: Session) -> None:
    if not _snapshot_matches_db(session):
        pytest.skip("DB was rebuilt since the committed snapshot; re-export to compare.")
    snap = load_scaffold_snapshot(SNAPSHOT)
    checked = 0
    for work in snap.works:
        entries = work.get_versions["indian"].get("versions", [])
        if len(entries) < 2:
            continue
        version_set = [uuid.UUID(str(e["version_id"])) for e in entries]
        combos: list[dict] = [{"language": entries[-1]["language"]}, {"era": "newer"}]
        if entries[0].get("year"):
            combos.append({"year": entries[0]["year"]})
        if entries[0].get("cast_lead"):
            combos.append({"actor": entries[0]["cast_lead"][0]})
        combos.append({"language": "bn"})  # empty-set path
        for by in combos:
            live = repository.refine_filter(
                session, version_set, repository.RefineBy.model_validate(by)
            ).model_dump(mode="json")
            local = refine_local(entries, by)

            def _sorted(payload: dict) -> list:
                return sorted(payload["versions"], key=lambda v: str(v["version_id"]))

            # Order-insensitive: the live tool's SQL `IN` carries no ORDER BY, so row
            # order follows Postgres physical order (changed by the 2026-07-04 restart);
            # the v0 contract promises set semantics, not order.
            assert _sorted(local) == _sorted(live), f"{work.work_key}: diverges for {by}"
            checked += 1
    assert checked >= 20
