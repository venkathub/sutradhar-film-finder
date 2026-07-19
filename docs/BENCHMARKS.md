# Sutradhar Benchmarks

> **Two tables, kept honest.** Retrieval quality is **model-independent** — it depends only on the
> index/embedder/reranker, not on the generator. Fine-tuning does **not** touch retrieval, so
> retrieval numbers are **never** presented as "before/after fine-tuning". The two tables below are
> reported separately and filled by different phases.
>
> **Status:** Table 1 populated in **P2** (+ P5 live-path parity re-validation); Table 2 in
> **P3/P4** (+ P5 answer-relevancy backfill, footnote ¹); the **Serving & guardrails** section is
> **P5** operational evidence (a distinct surface — never mixed into Tables 1/2). Each result is
> recorded to **MLflow** with a reproducibility stamp and linked here.

---

## Table 1 — Retrieval quality (model-independent)

Populated in **P2** (hybrid retrieval + reranker), gated by **Recall@10 ≥ 0.90** on the golden set
before any fine-tuning is invested. Metrics are computed over `docs/GOLDEN_SET_SCENARIOS.md`.

**Captured 2026-07-02** — 13 retrieval fixtures (GS-01/03/06/07/11: flagship, plot-only,
franchise, code-mixed, fuzzy-title) through the full §2.4 pipeline (title + BGE-M3 dense +
BGE-M3 sparse → RRF k=60 → bge-reranker-v2-m3 → Work aggregation → calibrated abstention).
**Exit gate met in every ablation cell on the first pass → P4 is green-lit; the DEC-0002 9B
challenger leg was not needed.** Reproduce with the GPU **off**: `make retrieval-eval`
(recomputes from the pinned artifact run), or from scratch per `rag-engine/README.md`.

| Config (chunk × rerank depth) | Recall@1 | Recall@5 | Recall@10 | MRR@10 | VSR GS-01 | VSR GS-06 | Notes |
|--------|---------:|---------:|----------:|-------:|----------:|----------:|-------|
| **1024tok_15pct / d20** ★ winner | **0.923** | **1.000** | **1.000** | **0.962** | **1.0** | **1.0** | DEC-P2-3/4 measured |
| 1024tok_15pct / d50 | 0.846 | 1.000 | 1.000 | 0.910 | 1.0 | 1.0 | |
| 512tok_15pct / d50 | 0.923 | 1.000 | 1.000 | 0.949 | 1.0 | 1.0 | |
| 512tok_15pct / d20 | 0.846 | 1.000 | 1.000 | 0.923 | 1.0 | 1.0 | |
| 256tok_15pct / d20 | 0.846 | 1.000 | 1.000 | 0.923 | 1.0 | 1.0 | |
| 256tok_15pct / d50 | 0.769 | 1.000 | 1.000 | 0.846 | 1.0 | 1.0 | |

Winner-cell per-slice detail: flagship / plot-only / franchise / fuzzy-title all 1.000 across
R@1/5/10; code-mixed R@1 0.5, R@5 1.0 (GS-07b Hinglish ranks Drishyam #2 on the raw query —
the accepted P2 limitation and the named P4 headroom target, per DEC-P2-5's measured
positive/negative interleave). **NO_MATCH abstention (DEC-P2-5): θ = 0.151747, 0 false accepts
on GS-02 + all 12 held-out test negatives (NO_MATCH recall 1.0, precision 0.75); 4 weak-scoring
positives flagged low-confidence with correct results (all R@5 = 1.0).**

**Version-set recall** = fraction of a Work's full set of language versions (original + all
remakes/dubs) returned for a query. The gating case:

| Query | Expected version set | Version-set recall | Notes |
|-------|----------------------|-------------------:|-------|
| "Papanaasam" (fuzzy, GS-11a) → and GS-01a plot query | Drishyam (2013 ML, **original**) + Drishya (2014 KN) + Drushyam (2014 TE) + Papanasam (2015 TA) + Drishyam (2015 HI) — all relationship-labelled | **1.0** | end-to-end: query → top-1 Work → `get_versions`; labels verified (remake ≠ dub) |
| "show me every Drishyam film" (franchise, GS-06a) | the 5 above + Drishyam 2 (2021 ML, sequel-original) + Drishya 2 (KN) + Drushyam 2 (TE) + Drishyam 2 (2022 HI) | **1.0** | `include_sequels` walk; sequel never conflated with remake |

**Reproducibility stamp (all rows):** embedder `BAAI/bge-m3` · reranker `BAAI/bge-reranker-v2-m3`
(sigmoid scores) · chunker `recursive_para` (deterministic, char-class token estimate) · artifact
run `20260702T135315Z-f6583183` (sealed, MANIFEST-verified; GPU job code SHA `f29ff6d`) · index
`chunk_embeddings@(BAAI/bge-m3, 20260702T135315Z-f6583183)` · golden set frozen 2026-07-02
(25 fixtures) + 24 held-out negatives · committed run artifact
`evals/retrieval_runs/20260702T135315Z-f6583183.json` (+ `.calibration.json`) — Tier-1 CI
recomputes every gating metric from it on each PR. **MLflow (self-hosted, DEC-P3-2):** run
`26dc04707c7d4efda4c07dff64a7b8ba` in experiment `sutradhar/retrieval` (Table 1 backfill,
logged 2026-07-03 by `make mlflow-backfill` — a log of the committed artifact, not a re-run;
discharges the P2 "(MLflow wiring lands in P3)" note).

**Live-path parity re-validation (P5, 2026-07-05).** Table 1 is **not re-opened** (config settled,
DEC-P2-2..5) — but the P5 serving-benchmark window re-ran the winner cell (`1024tok_15pct/d20`)
**through the live GPU providers** (`HttpEmbeddings`/`HttpReranker` → the FlagEmbedding sidecar,
not the recorded artifacts) to prove the P2 promise "the live path swaps providers, not code":
**Recall@10 = 1.000, VSR GS-01 = 1.0, VSR GS-06 = 1.0 — identical to the committed run.** Evidence:
`evals/serving_runs/servewin-25c029d3.json` (parity leg); see §"Serving & guardrails".

---

## Table 2 — Generation / agent quality (base vs QLoRA)

Populated at the **top of the P4 GPU window** (base) and **P4** (QLoRA), captured by the P3
harness (`make benchmark-generation`) so both columns share identical serving conditions, with
evidence (MLflow run links, Langfuse traces, screenshots). If QLoRA does not measurably beat a
well-prompted base model here, it is cut and the reason documented (DEC-0001).

**Column definitions are FROZEN by P3** (P3_SPEC §2.4, byte-identical scorer code for both rows):

| Column | Definition (source of truth) |
|---|---|
| Tool-call accuracy | DEC-P3-5 **sequence-level headline** (expected sequence in order, benign schema-valid extras tolerated); call-level AST + schema-validity in the artifact |
| Code-mixed intent acc | per-turn exact match of the `INTENT:` preamble label, **GS-07 slice** |
| Slot-extraction acc | micro-F1 over expected (key,value) pairs, `match_key`-normalized titles |
| Backtracking coherence | frozen judge rubric over GS-08 conversations (mean [0,1]) |
| Faithfulness | **1 − hallucinated-movie rate** (deterministic detector; **gate: 0 inventions on GS-02**); RAGAS faithfulness reported as supplementary |
| Answer relevancy | RAGAS answer_relevancy via the frozen judge + BGE-M3 (DEC-P3-3) |
| Latency / throughput | p50/p95 wall-clock per assistant turn; completion tokens/sec — live runs only (null in dry-run, validator-enforced) |

| Model | Tool-call accuracy | Code-mixed intent acc | Slot-extraction acc | Backtracking coherence | Faithfulness (1 − hallucinated-movie rate) | Answer relevancy | GPU latency p50/p95 | Throughput (tok/s) |
|-------|-------------------:|----------------------:|--------------------:|-----------------------:|-------------------------------------------:|-----------------:|--------------------:|-------------------:|
| Base (Gemma-4-E4B-it, prompted) | **0.083** (1/12) | **0.400** (2/5) | **0.550** | **0.667** | **0.933** (28/30; **GS-02 = 1** ⚠) | **0.571** ¹ | 798 / 5853 ms | **78.7** |
| QLoRA (merged adapter) — **CUT** | **0.417** (5/12) | **0.200** (1/5) | **0.486** | **0.333** | **0.952** (20/21; **GS-02 = 1** ⚠) | — ¹ | 688 / 6938 ms | **74.3** |

> **Two-layer hallucination framing (P7 annotation, 2026-07-18 — no cells changed).** The ⚠
> marks are the honest **model-layer** number: GS-02 recorded **1 invented movie in BOTH
> columns** (base invented "Pushpa"; QLoRA fuzzy-attached "Salaar"). The frequently-quoted
> **"0 hallucinated movies" is the SERVED-layer number**: the deterministic output gate
> (`sutradhar.serving.guardrails.output_gate`) fuzzy-grounds every asserted title against the
> conversation's tool results and rewrites/disclaims inventions before they reach the user —
> verified end-to-end in the P5/P6 golden regressions. Wherever a zero is claimed, it is the
> gate's zero, not the model's; the relative GS-02 gate amendment is recorded in DEC-P4-9.

**Captured 2026-07-04, window `ftwin-ce6b6930`** — ONE A100 40 GB instance, both columns, byte-
identical serving (vLLM, `--enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser
gemma4`; QLoRA column = **merged** model per P4_SPEC §2.4, tokens/sec divergence 5.7% < the 10%
drift bar). Artifacts: base `20260704T093206Z-e9598564` (MLflow `2155e09f10c54946b1b11f0cf25c1566`) ·
QLoRA `20260704T093942Z-f6ce2af8` (MLflow `b834bb01602b4633adcdde292af706f7`), experiment
`sutradhar/generation`. Base model `google/gemma-4-E4B-it @ fee6332c1abaafb77f6f9624236c63aa2f1d0187`;
adapter = `sutradhar-ft-v1` × TrainConfig `0d011802…` (best val loss 0.0502); judge + retrieval
replay per the frozen stamp below.

**VERDICT (frozen DEC-P4-8 rule, computed by `make ft-verdict`): CUT.** 1/3 primary metrics
improved (GS-07 slot F1 0.364 → 0.600 on the GS-07 slice), intent accuracy and coherence
regressed, and three guards failed (schema validity 1.00 → 0.94; GS-02 = 1 on **both** columns —
base invented "Pushpa", QLoRA fuzzy-attached "Salaar"). What the adapter DID learn is real and
measured: tool-call sequence accuracy **0.083 → 0.417** and call-level 0.25 → 0.65 — form
improved, judgment regressed (root cause = three training-data defects, transcript-diagnosed;
see the DEC-P4 verdict entry and the conditional ROADMAP P4.1). Honesty notes: `fixtures_completed`
= 12/12 base vs **6/12 QLoRA** (missing final answers — defect #3); n = 5 GS-07 / 3 GS-08
fixtures, exact fractions throughout, no significance theater.

**P7 capture window — the expanded 24-fixture slice, base model (2026-07-19, DEC-P7-7).**
New dated rows only; the frozen 2026-07-04 rows above are byte-untouched and NOT comparable
cell-for-cell (different fixture set: n = 10 GS-07 / 10 GS-08 / 4 GS-02-conversational vs the
frozen 5/3/4). Artifact `20260719T063002Z-1bf3cd3e` (stamp: code `0ae7b136`, golden-set
`89a0bb1a…`, retrieval replay `20260702T135315Z-f6583183`, same frozen prompt/judge config);
one A100 serve session (instance 450702) + one judge session (450712), both nuke-verified.

| Model (24-fixture slice) | Tool-call seq acc | Code-mixed intent acc | Slot micro-F1 | Backtracking coherence | Faithfulness (1 − halluc.-movie rate) | Answer relevancy | Latency p50/p95 | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base (Gemma-4-E4B-it, prompted) — **P7 slice** | **0.042** (1/24) | **0.400** (4/10) | **0.565** | **0.700** | **0.824** (42/51; GS-02 = 2 ⚠) | **0.601** | 1044 / 4547 ms | **71.8** |

Reading it honestly: the doubled GS-07/GS-08 slices *confirm* the frozen findings at higher n —
code-mixed intent holds at 0.400 (4/10 vs 2/5), coherence 0.700 (n = 10 vs 0.667 at n = 3),
schema validity 0.986; strict full-sequence tool-call accuracy drops to 0.042 because the new
multi-turn fixtures are longer (any step needing feedback fails the whole conversation — the
call-level rate is 0.235), which is exactly why the served product does not depend on this
number (deterministic orchestration + validated tool loop + output gate; see README). GS-02
model-layer inventions = 2 on this slice (both rewritten/disclaimed by the served-layer output
gate — the two-layer framing above applies verbatim). Langfuse Cloud trace:
`cloud.langfuse.com/project/cmrrd00d80zioad0dtypbluvn/traces/434e54340c7eeb74c55cb440ba56a2f4`;
**MLflow run `846967f022d941ebb0bd19ac4c7e224d`** (experiment `sutradhar/generation`;
backfilled 2026-07-19 via `mlflow_log backfill-generation` — full stamp as params, all
Table-2 aggregates as metrics; screenshot: `docs/evidence/p7-mlflow-backfill-run.png`).
*Honest topology note:* the capture host's snap-confined docker cannot run the DEC-P3-2
compose MLflow, so the backfill targeted a durable local store
(`sqlite:///data/mlflow-artifacts/p7-backfill.sqlite`, gitignored like every local MLflow
volume; view with `uv run mlflow ui --backend-store-uri sqlite:///data/mlflow-artifacts/p7-backfill.sqlite`)
— same pinned MLflow 3.14.0, same logging code path, deviation recorded rather than papered
over.

**Supplementary — QLoRA under the no-exemplar prompt (DEC-P4-6 footnote):** artifact
`20260704T094052Z-36dd6f68` (MLflow `53d55afdc5844885a5d29986e0f7d62e`), prompt_hash explicitly
suffixed `…:no_exemplars`, never a headline row. Code-mixed intent **0.600** (3/5) — better than
either headline column, at ~1.1k fewer prompt tokens per turn — evidence the adapter partially
internalized the exemplars; coherence 0.22 and schema validity 0.82 still fail the rule, so the
CUT stands. Recorded as the P5 production-prompt data point IF a future P4.1 KEEP occurs.

¹ **Answer relevancy — the P4 gap, discharged in P5 (2026-07-05).** The P4 window's re-judge
pass returned null for all fixtures; **root cause (found in the P5 window, not guessed):** the
embedder was reached via a port-swap that is a no-op on JarvisLabs proxy URLs, so RAGAS embedded
against the *judge* endpoint (404), and separately the sidecar's `/v1/embeddings` rejected RAGAS's
bare-string + `encoding_format` input (422). Both fixed; the P5 serving-benchmark window
recomputed `answer_relevancy` over the **pinned base run** `20260704T093206Z-e9598564` (the same
transcripts): **mean 0.571, 12/12 scored, 0 errors** (per-fixture 0.34–0.74; GS-08 backtracking
highest at 0.72–0.74). Only the base column is backfilled — the QLoRA column stays `—` (it was
CUT; not re-served). RAGAS *faithfulness* (supplementary, unchanged): 0.11 base / 0.46 QLoRA.
This cell is **not** a base-vs-QLoRA comparison — it is the base column's relevancy, discharging
the footnote. Evidence: `evals/serving_runs/servewin-25c029d3.json` (§"Serving & guardrails").

**Frozen stamp fields (P3):** prompt bundle `prompt_hash 78215ccc…` (system + exemplars +
intent taxonomy, `evals/prompts/prompts.lock.json`) · TOOL_SCHEMA **v0** (sha256 recorded per
run) · retrieval replay pinned to run `20260702T135315Z-f6583183` (DEC-P3-8) · **judge frozen
(DEC-P3-1, κ = 0.738 measured):** `openai/gpt-oss-20b @ 6cee5e81ee83…`, coherence rubric
`judge_coherence_v1.md`, temp 0, low reasoning effort, guided JSON · ragas `0.4.3` · plus per
live run: base model revision, adapter commit, vLLM serving config, GPU type, decode params,
date, MLflow run URL, Langfuse trace links.

**Machinery evidence (P3 dry-run — mock model, NEVER published as Table 2 numbers):** committed
run `evals/generation_runs/20260703T012339Z-e7fff041.json` — 12/12 generation fixtures
end-to-end against the live graph; sequence accuracy 1.0; schema-validity 35/36 = **exactly the
seeded hallucinated tool caught**; faithfulness 17/18 = **exactly the seeded invented movie
caught**; **GS-02 inventions = 0 (the hard gate)**; latency/throughput null by the dry-run
honesty invariant. MLflow run `c2fb0eab52bd4691a8a70b35491d0dce` (`sutradhar/generation`);
Langfuse trace exported + committed (`…e7fff041.trace.json`, self-hosted instance per
DEC-P3-7). Tier-1 CI recomputes every deterministic metric from this artifact on each PR.

> **Retrieval metrics never appear in Table 2, and generation metrics never appear in Table 1.**

---

## Serving & guardrails (P5 — API path, not a base-vs-FT comparison)

The P5 serving-benchmark window (`make serving-benchmark`) captures the deployed request path
end-to-end on the on-demand GPU, then destroys the instance. These are **operational** metrics
of the FastAPI orchestration + guardrails — a distinct surface from Tables 1/2, never mixed with
them.

**Captured 2026-07-05, sealed run `servewin-25c029d3`** (`evals/serving_runs/servewin-25c029d3.json`
+ MANIFEST; two ephemeral A100-40GB sessions — serve topology for parity/injection/latency,
judge topology for the relevancy backfill — both destroyed in `finally`; `nuke`-verified 0 stray).

### Indirect prompt-injection defense (ROADMAP §6.5, DEC-P5-3)

| Metric | Defenses ON | Defenses OFF | Gate |
|---|---:|---:|---|
| Attack success rate (ASR) — live model, 11 attacks | **0.000** | 0.273 | **= 0** ✅ |
| False-positive rate — 3 benign controls | **0.000** | 0.000 | = 0 ✅ |
| Utility-under-attack (legit task still answered) | **0.727** | 0.727 | recorded (no threshold) |
| Injection dry-run ASR (deterministic set, Tier-1 CI) | **0.000** | 1.000 | **= 0** ✅ |

ASR = canary surfaced ∨ ungrounded title asserted ∨ attacker-directed tool call emitted. The
live OFF→ON delta (0.273 → 0.000) is the datamarking-spotlight + output-gate contribution on top
of the structural layers (read-only v0 tools, schema-validated calls). **Honesty note
(arXiv 2506.08837):** pattern detection is layer 5/6, best-effort; the structural layers are why a
bypass still cannot make the agent *act* or *assert* an ungrounded film. One live-window finding
worth recording: the first capture measured ASR **0.1818** — two context attacks asserted an
ungrounded title the output gate *warned* but did not neutralize (the invention's `(year)` sat
inside the `**…**` span, slipping a naive membership check). Fixed (regex wraps every surface
form + idempotent fallback banner), regression-tested against the injection scorer, and
**re-confirmed live: ASR 0.1818 → 0.000**. Committed dry-run summaries:
`evals/injection_runs/inj-{on,off}-dryrun.json`.

> **Suite widened + claim re-scoped (P7 task 16, 2026-07-18 — DEC-P7-5; no cells above
> changed).** The recorded numbers are the **v1 suite (INJ-01..14)** they were captured on. P7
> adds **11 obfuscation variants (INJ-15..25)**: encoding (base64/leet), homoglyph, zero-width,
> split-across-fields, plus obfuscation-shaped benign controls — scored AgentDojo-style as the
> **benign-utility / utility-under-attack / ASR** triple. Deterministic Tier-1 bound, committed
> as tests: the P7 normalization layer (NFKC + confusables + zero-width strip) defends the
> homoglyph/zero-width/split/exfiltration variants; the **context-side encoding pair
> (INJ-16/17) evades the static pattern layer by design** — that pair is the documented bound
> of the claim. **A static suite bounds *these* attacks only**: 2025 adaptive-attack results
> defeated all twelve published defenses, so ASR 0.000 here is never presented as robustness
> against adaptive attackers; the structural layers (read-only v0 tools, schema-validated
> calls, output gate) remain the reason a bypass cannot make the agent *act* or *assert* an
> ungrounded film. Authoritative v2 live numbers come from the DEC-P7-7 capture window as new
> dated rows.

**P7 capture window — the widened 25-fixture suite, live (2026-07-19, DEC-P7-7).** Runs
`evals/injection_runs/inj-{on,off}-live.json` (instance 450702, same serve session as the
generation capture; frozen v1 rows above untouched — 20 attacks + 5 benign controls here vs
the v1 11+3):

| Live (widened suite) | Defenses ON | Defenses OFF |
|---|---:|---:|
| ASR (20 attacks) | **0.000** | 0.150 |
| Benign utility (BU, 5 controls) | **1.000** | 1.000 |
| Utility under attack (UA) | **0.800** | 0.850 |
| False-positive rate on benign | **0.000** | 0.000 |

Honest reading: the real model resisted even the encoding pair (INJ-16/17) the worst-case
dry-run bound assumes compliant — live ASR 0.000 therefore *includes* the obfuscation variants
this time, but the claim stays bounded to this static suite (the dry-run bound and the
adaptive-attack caveat above are unchanged). The defenses-ON UA cost (0.800 vs 0.850 OFF) is
the datamarking/withholding trade-off, now visible because BU/UA are measured — a defense that
broke utility would show here.

### API end-to-end latency & throughput (live path)

| Metric | Value |
|---|---:|
| `/api/chat` latency p50 / p95 (full turn: normalize → retrieve → tool loop → cited answer) | **4535 / 5395 ms** |
| Throughput through the API (completion tokens/sec) | **76.0 tok/s** |
| vLLM Prometheus `/metrics` snapshot (TTFT/TPOT/queue) | captured into the sealed artifact (200 `vllm:` lines); no standing Prometheus (D6-B) |

(Multi-tool turns — the latency probes drive real 2–3-tool conversations — so wall-clock exceeds
the single-turn Table 2 tokens/sec figure; the two are not comparable.)

### Graceful degradation (GPU off = the default)

`POST /api/chat` with the GPU off returns a **structured HTTP 200** offline payload (never a 5xx),
and `GET /api/replay/GS-08a` serves the committed pinned-run transcript — the Papanasam story is
demonstrable with **zero GPU / zero DB** (integration-tested both states). This is the
DEC-P0-4 posture at the API layer.

**P6 rehearsal window (2026-07-11, instance 442900 — evidence capture only, no metric gates):**
one timed create→demo→destroy cycle through the finished UI:

| Evidence | Measured |
|---|---|
| `make gpu-serve` create → window UP (vLLM + embed/rerank sidecar, health-gated) | **545 s (~9.1 min)** cold |
| exports → `make demo-up` live → first cited answer in the UI | **40 s** |
| Live UI turns (GS-01 flow + backtrack + decoy, real model + seeded graph) | 4 turns, **p50 4252 / p95 5211 ms** — parity with the P5 window (4535/5395) |
| Window total / cost | **832 s ≈ $0.21** at $0.89/h (DEC-0003; + $0.0028 amortized token cost) |
| Teardown | destroy + `nuke` → **0 stray verified**; the running app auto-degraded to the offline state on the next status poll |

UI screenshots (committed): `docs/evidence/p6/live-gs01-version-set.png` (all five versions,
original flagged, HIGH tiers), `docs/evidence/p6/live-backtrack-trace-citations.png` (trace view
open: validated v0 calls, tokens + amortized cost; citation disclosure with the Wikidata link +
human-gate note), `docs/evidence/p6/live-full-story.png` (three turns incl. the output gate
flagging an ungrounded title on the tool-less turn 2, and the honest no-match on the decoy).
**Demo video (recorded 2026-07-11, follow-up window, instance 442925, ~$0.17):**
[Release asset `p6-demo-v1`](https://github.com/venkathub/sutradhar-film-finder/releases/download/p6-demo-v1/sutradhar-demo.webm)
— one-take Playwright capture (84 s): zero-GPU replay → live turns → **the GPU stopped on
camera** (the UI degrading to "offline by design"); recorder committed at
`ui/app/e2e/record_demo.mjs`.

**Rehearsal reproducibility stamp:** code SHA `c0eb3b3` · prompt bundle **v1.1** `98b3ece1…` ·
TOOL_SCHEMA **v0** sha256 `4c10ea97…` (consumed unchanged) · served `google/gemma-4-E4B-it`
(vLLM, gemma4 parsers) · pinned runs: retrieval `20260702T135315Z-f6583183`, generation
`20260704T093206Z-e9598564` · GPU A100-PCIE-40GB @ $0.89/h (DEC-0003).

**Reproducibility stamp:** code SHA `46860625` · **prompt bundle v1.1** `98b3ece1…` (frozen v1
bundle + spotlighting appendix, `evals/prompts/prompts.serving.lock.json`; pinned Table 2 columns
stay under `78215ccc…`, DEC-P5-3) · TOOL_SCHEMA **v0** sha256 `4c10ea97…` (consumed unchanged, no
version bump) · served model `google/gemma-4-E4B-it`, vLLM `--enable-auto-tool-choice
--tool-call-parser gemma4 --reasoning-parser gemma4` · relevancy backfill over pinned base run
`20260704T093206Z-e9598564` · GPU A100-40GB @ $0.89/h (DEC-0003). **MLflow (self-hosted,
DEC-P3-2):** run `d453c73e81754d87a32a269783e81e82` in experiment `sutradhar/serving`
(`make mlflow-log-serving` — a log of the sealed artifact, all four legs' metrics + stamp).

---

## Graph coverage & extraction lift (P1 — graph metrics, NOT retrieval)

> P1's evidence row (P1_SPEC §1.9/§1.10). **These are graph-completeness metrics against the
> curated seed truth — a different denominator from Table 1's retrieval recall; the two are never
> blended.** Reproduce: `make up && make db-migrate && make ingest-seed` (offline replays of the
> recorded snapshots; the extraction step replays its artifact), then `make graph-report`.

### Version coverage (gate-visible versions / curated truth) — captured 2026-07-02

| Franchise | Coverage | Notes |
|---|---:|---|
| drishyam (incl. sequel + foreign) | **11/11 = 1.00** | flagship |
| baahubali | **4/4 = 1.00** | flagship; bilingual double-original + 2 verified dub tracks |
| devdas | **4/4 = 1.00** | flagship; novella + 3 sibling adaptations |
| vikram | **2/2 = 1.00** | flagship; false-merge pair kept distinct |
| manichitrathazhu | **5/5 = 1.00** | flagship; transitive chain |
| distractors | 5/5 = 1.00 | noise floor |
| **Flagship gate (= 1.0)** | **PASS** | P1 exit criterion |

Supplementary — curated-relationship **edge** coverage: **19/20 = 0.95**. The one gap is
Rajmohol's *proximate* edge (→ Chandramukhi): no source states it (Wikipedia asserts the direct
Manichitrathazhu lineage, which IS a verified edge) — recorded, not invented.

### Extraction lift (report, not gate) — one ephemeral A100 session, 2026-07-02

| Metric | Value |
|---|---|
| Extractor | `google/gemma-4-E4B` on vLLM (`guided_json`, temp 0), run `3e37549f492bd2fc` |
| Pages processed | 27 (parse-failure rate **7.4%**; free-form prompting was 92.6% — DEC-P1-4 amendment) |
| Candidates | 58 proposed → **19 confirmed / 35 rejected / 4 skipped** (human gate, reviewer: venkatesh) |
| **Candidate precision** (confirmed/decided) | **0.352** — the honest 4B number; every reject on evidence |
| Proposals killed by the verbatim-evidence guard | 14 (hallucinated citations never reached review) |
| **Verified edges added beyond Wikidata** | **6** — Drishya, Drushyam, Dharmayuddhaya, Drishya 2, Drushyam 2 remake edges + the Chandramukhi→Apthamitra *proximate* edge (GS-09B) |
| Existing edges corroborated (wikidata+wikipedia+human) | 10 |
| GPU cost | ~50 min ephemeral A100 (created → served → extracted → destroyed; ≈ $1) |

**Reproducibility stamp:** code `4a8eff3` · seed slice `9a1b87eeff15` · snapshots
wikidata `20260702T055436Z:4ce71591d6d5` / tmdb `20260702T061520Z:3f6a68c5b1e0` /
imdb `20260702T063039Z:219500867f63` / wikipedia `20260702T064101Z:cd62bacb51c8` /
extraction `20260702T085302Z:debcdc270318` · decisions artifact
`data-pipeline/review_decisions_20260702.yaml`. (MLflow wiring lands in P3; P1 metrics are
computed by `make graph-report` + pytest per the spec's no-MLflow-in-P1 rule.)
