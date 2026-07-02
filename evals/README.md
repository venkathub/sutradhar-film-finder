# evals

Evaluation harnesses for Sutradhar. **Import package:** `sutradhar.evals`

## Golden set (P1 — frozen)

`evals/golden/*.yaml` holds **25 fixtures across all 11 scenario categories**
(GS-01..GS-11 per `docs/GOLDEN_SET_SCENARIOS.md`), authored only from verified graph facts
(each `verify_source` names the QIDs / pinned pages). `make golden-validate` runs the
validator against the live graph:

- **schema** (pydantic; `expected_tool_calls` on GS-07/GS-08 validate against TOOL_SCHEMA v0);
- **golden eligibility** (DEC-P1-7 layer 3): every expected version/relationship must be
  gate-visible AND HIGH-or-human-verified — MEDIUM-backed or conflict-hidden fixtures are
  rejected (tested by mutation);
- **NO_MATCH proofs**: GS-02 negatives verify nothing in the graph resolves for them;
- category completeness (all 11 GS categories present).

CI (Tier-1 integration job) rebuilds the reviewed graph from recorded fixtures — including a
CI-mirrored review pass with the same semantics as the committed
`data-pipeline/review_decisions_20260702.yaml` — and re-validates every fixture plus the named
regressions (`test_gs01_version_set_recall`, `test_gs04_dub_vs_remake`, …).

Frozen 2026-07-02 against graph state: 15 works / 31 versions / 21 gate-visible edges
(reproducibility stamp in `docs/BENCHMARKS.md`).

## Planned (P2/P3)
- RAGAS harness + retrieval metrics (Recall@k / MRR / version-set recall) over these fixtures.
- The two-table benchmark runner (base vs QLoRA) reusing `expected_tool_calls`.
