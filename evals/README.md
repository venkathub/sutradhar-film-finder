# evals

Evaluation harnesses for Sutradhar. **Import package:** `sutradhar.evals`

## Golden set (P1 — frozen)

`evals/golden/*.yaml` holds **34 fixtures across all 11 scenario categories**
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

### Generation slice (P3 task 4 — the Table 2 fixtures)

12 fixtures carry the additional generation labels (`expected_intent`, `expected_slots` —
one entry per turn for multi-turn; labels drawn from the frozen taxonomy, test-enforced by
`tests/test_golden_generation_labels.py`) plus `expected_tool_calls` validated against
TOOL_SCHEMA v0:

| Slice | Fixtures | Notes |
|---|---|---|
| GS-07 (code-mixed) | a Tanglish, b Hinglish (plot); **c Kanglish, d Tenglish (+actor slot), e native-script Kannada** (title path) | intent/slot accuracy |
| GS-08 (backtracking) | a Drishyam; **b Manichitrathazhu franchise; c Drishyam 2 with a mid-turn NO_MATCH** (no Tamil version exists — abstain without losing the set) | backtracking coherence |
| GS-02-conversational | **d Hinglish title (Salaar), e Hinglish plot, f Tanglish plot+language, g native-Telugu title (Pushpa)** | 0-inventions gate, end-to-end |

GS-02a/b/c stay retrieval-shaped and untouched (the P2 committed artifact keeps validating);
all new queries were verified against the live title index before freezing (negatives resolve
to nothing — including their extracted title forms; positives resolve at score 1.0). All 34
fixtures pass `make golden-validate` (expanded 2026-07-03, same graph state).

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

## Frozen prompt artifacts (P3 — DEC-P3-4)

`evals/prompts/` holds the **hash-pinned base-model prompting strategy** the generation benchmark
runs under (loader: `sutradhar.evals.prompts`):

- `system_v1.md` — system prompt: grounding rules (assert only tool-returned films; cite;
  label all five relationship types; flag the original; abstain out-of-catalog), tool-usage
  and multi-turn behaviour, the **INTENT preamble contract**: every final prose answer per
  user turn begins with one machine-read line `INTENT: {"intent": …, "slots": …}` — the source
  the intent/slot-accuracy scorers parse (placement rationale in DEC-P3-4's 2026-07-03
  amendment) — and the **bold-title formatting contract**: every asserted film title in
  `**bold**`, nothing else bold — the deterministic surface the no-hallucinated-movie
  detector extracts (task-5 amendment).
- `exemplars_v1.md` — 3 handcrafted few-shot exemplars (one code-mixed Hinglish plot query, one
  multi-turn refine, one NO_MATCH abstention), appended to the system message as one hashed unit.
  Franchises (Ghajini, Okkadu/Ghilli, Interstellar) are deliberately **outside the golden set**;
  disjointness is test-enforced (no fixture query substring may appear in the bundle).
- `intent_taxonomy_v1.json` — the frozen 6-label taxonomy (`find_by_plot | find_by_title |
  list_versions | refine | disambiguate | out_of_catalog`) + 7 slot keys (= `refine_filter.by`
  vocabulary + `plot_description` + `title`) + the preamble format.
- `prompts.lock.json` — pinned SHA-256 per file + the combined **`prompt_hash`** stamped into
  every generation-run artifact and the Table 2 stamp. Any edit fails
  `tests/test_prompt_artifacts.py` until deliberately re-pinned:
  `uv run python -m sutradhar.evals.prompts --write-lock`. The P4 QLoRA before/after is only
  comparable under an identical `prompt_hash`.

## Generation metric scorers (P3 task 5 — `sutradhar.evals.generation`)

Pure, laptop/CI-safe scoring functions (string math only; the task-6 driver produces
transcripts, these score them, Tier-1 re-runs the same bytes over the committed artifact):

- **Tool-call accuracy (DEC-P3-5):** call-level AST match (name + placeholder-bound,
  normalized args; free-text `description` compared by token overlap, titles by `match_key`)
  + sequence-level headline (expected sequence in order; benign schema-valid extras
  tolerated) + schema-validity as the independent third number. `$work_id`/`[$version_set]`
  bind only to ids earlier tool results actually returned — an unseen id is a scored
  mismatch, never a crash.
- **Intent accuracy:** per-turn exact match of the parsed `INTENT:` preamble label
  (missing/malformed preamble = wrong, never an exception).
- **Slot accuracy:** micro-F1 over (key, value) pairs; titles normalized via `match_key`
  (native-script ↔ romanized matches), everything else casefolded.
- **No-hallucinated-movie detector (the GS-02 gate):** asserted titles (bold spans per the
  frozen contract + an unbolded `Title (year)` fallback with a language/meta-word guard)
  must fuzzy-resolve (≥ 0.80, DEC-P1-5) to a tool-result title of the same conversation;
  "Papanaasam"-style variants pass, inventions are counted; abstentions assert nothing.

## Conversation driver (P3 task 6 — `sutradhar.evals.driver`)

The multi-turn validate→execute→feedback loop that runs one golden fixture conversation
against any OpenAI-compatible endpoint (`LLMClient.chat()`; mock in dry-run, vLLM in the
P4 GPU window):

- **Outbound tools are generated from `tool_schema.v0.json`** — never hand-written, so
  drift from the frozen schema is impossible (P3_SPEC §2.8).
- **Every emitted call is validated BEFORE execution** (the DEC-P1-8 jsonschema validator
  applied to model output): hallucinated tools, hallucinated parameters and wrong-typed
  arguments are recorded as scored verdicts, the error is fed back as the tool result, and
  the loop continues — bounded by `MAX_TOOL_ROUNDS = 6` per turn, never a crash.
- **Valid calls execute against `sutradhar.graph.repository`** (the five frozen v0 tools).
  `search_by_plot` **replays the committed P2 retrieval run** (pinned `RETRIEVAL_RUN`,
  per-fixture records): no neural op on the laptop, and both Table 2 columns see
  byte-identical tool behaviour; fixtures absent from the recorded run (the P3
  conversational negatives, out-of-catalog by construction) replay as honest abstention.
  Description *quality* is still scored by tool-call accuracy.
- **Full transcript capture** (`FixtureTranscript`): every message, call (raw + verdict +
  bound arguments + result/error), per-round usage and latency, per-turn final answers —
  what the task-9 artifact commits and Langfuse traces mirror. Endpoint off/error →
  `chat_status` recorded, fixture aborts gracefully (DEC-P0-4).

Integration proof (`tests/integration/test_driver_e2e.py`): GS-08a end-to-end against the
live Postgres graph with a scripted mock model that binds REAL ids from prior tool results —
all five v0 tools exercised, every result v0-shape-valid, DEC-P3-5 sequence score 1.0 with
benign extras tolerated, zero hallucinated movies.

## LLM-as-judge (P3 task 7 — `sutradhar.evals.judge`, DEC-P3-1)

Self-hosted OSS judge (gpt-oss-20b primary, Phi-4-14B alternate) served by vLLM in a
**short ephemeral GPU session** (`make gpu-judge`: serve `JUDGE_MODEL` + BGE-M3 → run the
κ report → destroy; teardown guaranteed). The client is OpenAI-compatible and env-driven
(`JUDGE_BASE_URL`/`JUDGE_MODEL`/`JUDGE_API_KEY`), so the DEC-P3-1 self-hosted ↔ frontier
escalation is pure config — proven by test against both endpoint styles.

- **Governance pins:** rubric prompts live in-repo (`evals/prompts/judge_coherence_v1.md`,
  `judge_faithfulness_v1.md`) and hash into `JudgeConfig` (separate from the base-model
  `prompt_hash`); temperature 0; guided decoding + pinned low reasoning effort on the wire
  (gpt-oss-20b is a reasoning model). Malformed judge output → `judge_error` recorded,
  never a crash; a judge_error binarizes to the negative class so it *hurts* agreement
  rather than hiding.
- **Human-agreement validation** (`sutradhar.evals.judge_validation`, blocks the judge
  freeze — task 13): `make judge-worksheet` builds the ~24-item blind worksheet from
  committed transcripts (+ deterministic foils: a re-answer-turn-1 coherence foil and an
  invented-movie faithfulness foil; provenance in a separate key file), the human fills
  every `human_label`, `make judge-validate` scores it with the pinned judge and reports
  percent agreement + **Cohen's κ (gate: ≥ 0.6)** → `evals/judge_validation/report.json`.
  Judge–human agreement result: _pending task 13 (the labelling + ephemeral GPU session)_.

## Planned (P3+)
- Retrieval eval harness + metrics — **landed in P2** (see above).
- Frozen prompt artifacts — **landed in P3 task 3** (see above).
- RAGAS harness; MLflow/Langfuse wiring; the two-table benchmark runner (base vs QLoRA)
  reusing `expected_tool_calls` (P3+).
