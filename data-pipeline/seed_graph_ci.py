"""Seed the graph from RECORDED fixtures — offline, deterministic (P3 task 12).

The same chain the integration suite proves (`tests/integration/test_golden_fixtures.py`
``reviewed``), run once and COMMITTED: Wikidata spine + TMDB enrich + titles + build_graph
+ candidate load + the CI-mirrored review pass. No network, no API keys — this is what
lets a fresh clone (and the Tier-2 dry-run job) reach a golden-validator-clean graph with
`make up db-migrate seed-graph-ci`. The full network ingestion (`make ingest-seed`) remains
the authoritative path for refreshing data; this replays the recorded state.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # tests.integration.ci_review_pass import

from sqlalchemy import select, text  # noqa: E402
from tests.integration.ci_review_pass import apply_ci_review_pass  # noqa: E402

from sutradhar.graph.db import create_graph_engine, create_session_factory  # noqa: E402
from sutradhar.graph.models import SourceId, SourceRef  # noqa: E402
from sutradhar.graph.schema import Version  # noqa: E402
from sutradhar.pipeline.build import build_graph  # noqa: E402
from sutradhar.pipeline.extract import load_candidates  # noqa: E402
from sutradhar.pipeline.seed import DEFAULT_SEED_PATH, load_seed_slice  # noqa: E402
from sutradhar.pipeline.titles import upsert_version_title  # noqa: E402
from sutradhar.pipeline.tmdb import enrich_tmdb, parse_movie  # noqa: E402
from sutradhar.pipeline.wikidata import ingest_spine, parse_entity  # noqa: E402

FIXTURES = REPO_ROOT / "tests" / "fixtures"

_TABLES = (
    "chunk_embeddings",
    "chunks",
    "candidate_edges",
    "edges",
    "conflicts",
    "plot_texts",
    "version_cast",
    "version_title",
    "version",
    "person",
    "work",
)


def main() -> int:
    engine = create_graph_engine()
    factory = create_session_factory(engine)
    with factory() as session:
        existing = session.execute(text("SELECT count(*) FROM work")).scalar_one()
        if existing:
            print(f"graph already has {existing} works — wiping and reseeding from fixtures")
        for table in _TABLES:
            session.execute(text(f"DELETE FROM {table}"))

        slice_ = load_seed_slice(REPO_ROOT / DEFAULT_SEED_PATH)
        wd = json.loads((FIXTURES / "wikidata" / "entities_sample.json").read_text("utf-8"))
        ingest_spine(session, slice_, {q: parse_entity(e) for q, e in wd.items()})
        tm = json.loads((FIXTURES / "tmdb" / "movies_sample.json").read_text("utf-8"))
        enrich_tmdb(session, {int(k): parse_movie(v) for k, v in tm.items()})
        for v in session.scalars(select(Version)).all():
            upsert_version_title(
                session,
                v.version_id,
                v.title,
                "canonical",
                v.language,
                [SourceRef(source=SourceId.HUMAN, ref="seed_slice")],
            )
        build_graph(session)
        art = json.loads((FIXTURES / "extraction" / "outputs_sample.json").read_text("utf-8"))
        load_candidates(session, art["raw_outputs"], art["pages"], art["model_id"])
        confirmed = apply_ci_review_pass(session)
        session.commit()

        works = session.execute(text("SELECT count(*) FROM ground_truth_works")).scalar_one()
        versions = session.execute(text("SELECT count(*) FROM ground_truth_versions")).scalar_one()
        print(f"seeded: {works} gate-visible works / {versions} versions; {confirmed} confirmed")
    engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(main())
