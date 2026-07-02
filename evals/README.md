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

## Held-out negatives (P2 — abstention calibration)

`evals/negatives/heldout.yaml` holds **24 NO_MATCH queries** (12 out-of-catalog plot
descriptions incl. code-mixed/native-script registers + 12 real-but-uncatalogued titles),
split 50/50 `calibration`/`test` (6 plot + 6 title in each half). **Import:**
`sutradhar.evals.negatives`.

These are deliberately **not** golden fixtures: they exist to tune the NO_MATCH abstention
threshold θ (DEC-P2-5) — θ is tuned on the calibration half only; NO_MATCH precision/recall
is *reported* on the test half. Tuning on GS-02 would contaminate the golden gate, so the
two sets never mix.

"Absent from slice by construction" is **enforced, not asserted**: the integration test
(`tests/integration/test_negatives_absent.py`) rebuilds the full title index (canonical +
TMDB + IMDb AKAs) and requires `resolve_title` to return zero candidates for every negative
at the rapidfuzz 0.80 radius (DEC-P1-5). A collision re-authors the negative, never moves
the threshold.

## Retrieval eval harness + committed run artifacts (P2)

`sutradhar.evals.retrieval` (metrics + ablation runner) and `sutradhar.evals.calibration`
(NO_MATCH θ, DEC-P2-5) — entrypoints `make retrieval-eval` / `make calibrate-no-match`.
`evals/retrieval_runs/` holds the **committed** per-run evidence (DEC-P2-6): the compact
eval artifact (per-query ranked works + abstention signals for every ablation cell),
the calibration report/curve, and the GPU-session record. Tier-1 CI
(`tests/test_golden_retrieval_regressions.py`) recomputes every gating metric from these
files on each PR — no DB, no GPU; a regression blocks merge. Results: `docs/BENCHMARKS.md`
Table 1 (P2 exit gate met: Recall@10 = 1.000, version-set recall = 1.0).

## Planned (P3+)
- Retrieval eval harness + metrics — **landed in P2** (see above).
- RAGAS harness; MLflow/Langfuse wiring; the two-table benchmark runner (base vs QLoRA)
  reusing `expected_tool_calls` (P3+).
