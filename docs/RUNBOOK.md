# Sutradhar Runbook — demo paths, timings, teardown

> **P6 task 10/11.** Three rehearsed paths: the zero-GPU demo (the default), the timed live
> interview demo (risk R4 — **rehearsed 2026-07-11**, timings measured below), and
> rebuild-from-scratch. **Nothing inference-side runs 24/7** (CLAUDE.md): every live path
> below ends with a verified teardown. All timings are measurements, never estimates
> dressed as measurements.

---

## Path A — Zero-GPU demo (the default; the 30-second path)

No GPU, no secrets, no model. Works from a fresh clone (CI-proven by the tier-1
`demo-smoke` job).

```bash
make demo-up          # .env from template -> compose (postgres, redis, app) ->
                      # migrate -> seed graph from recorded fixtures
# open http://localhost:8080/
```

What you see and say:

1. The UI comes up in **degradation mode** — the status pill reads *offline by design*;
   the offline notice explains the cost posture (this is a feature, not an outage).
2. Open the **replay browser** → `GS-08a`. The pinned benchmark transcript plays the
   three-turn backtracking story — *"the Drishyam with Ajay Devgn" → "no, the original
   one" → "is there a Telugu one?"* — with version cards (original flagged), per-claim
   citations, **real recorded GPU latencies**, and the tool-call **trace view**
   (schema-validated v0 calls, generated labels).
3. Talk over the [benchmark report](BENCHMARKS.md): retrieval Table 1 (Recall@10 = 1.0
   on the golden set, VSR 1.0 incl. the Papanasam/Drishyam case), Table 2 (base vs
   QLoRA — the measured **CUT** verdict), injection ASR 0 → the standing evidence *is*
   the proof.

Target: **< 1 min** from a warm laptop (image built once); first-ever run adds the image
build. Warm-laptop bring-up measured in the 2026-07-11 rehearsal: **25 s**
(`demo-up` end to end: compose `--wait` + migrate + reseed → UI 200).

Teardown: `make demo-down` (pgdata volume kept).

> Snap-confined Docker + repo outside `$HOME`: stage the repo first (see
> `infra/README.md` § "Snap-confined Docker workaround") and run `make demo-up` from the
> staged copy — the workaround applies to this path unchanged.

## Path B — Live interview demo (R4: on-demand GPU, timed, then STOPPED)

**Pre-call (do this before the meeting):** `.env` carries `JARVISLABS_API_KEY`; laptop
has the repo + built image (run Path A once). For live **plot search** the embedding
index must be loaded once: `make load-index` (reads the pinned artifact from
`data/artifacts/retrieval/` — pull via the DEC-P2-7 HF relay / `HF_ARTIFACT_REPO` if
absent); without it the title/graph tools still answer and `search_by_plot` degrades to
tool-error feedback. Budget: one demo cycle ≈ 15–25 min of A100-40GB @ **$0.89/h
(DEC-0003)** ≈ **$0.25–0.40**.

```bash
# 1. Bring up the serve window (ephemeral: create -> vLLM + embed/rerank sidecar ->
#    health-gate -> exports printed -> holds SERVE_HOLD_MINUTES -> destroy in finally)
make gpu-serve
#    ... prints when UP:
#    export LLM_BASE_URL=https://<instance>.notebooksn.jarvislabs.net/v1
#    export EMBED_BASE_URL=<sidecar>/v1
#    export RERANK_BASE_URL=<sidecar>/v1

# 2. In the API shell: export those three + the pinned retrieval artifact, then
export RETRIEVAL_RUN=20260702T135315Z-f6583183   # the pinned Table 1 winner artifact
# P7 (DEC-P7-2): the live chat path requires auth — mint token(s) for this window.
# The GPU-up /api/chat is the endpoint that burns GPU seconds; it is never open.
export API_AUTH_TOKENS="$(openssl rand -hex 16)"   # comma-separate for guests
echo "demo URL: http://localhost:8080/?token=${API_AUTH_TOKENS%%,*}"
make demo-up                                # same command — compose passes the env through;
                                            # the flip is exports, never a rebuild

# 2b. Open the printed ?token= URL — the UI adopts the token into sessionStorage
#     and strips it from the address bar. Unauthed visitors still get the full
#     GPU-off surface (replays, evidence); live turns 401 without the token and
#     are rate-limited (CHAT_RATE_LIMIT, token-first keying) with it.

# 3. Demo script (the gating story, live):
#    - Tanglish/Hinglish plot query -> cited answer
#    - "which movie is Papanasam a remake of?" -> ALL language versions, original flagged
#    - GS-02 decoy (e.g. "Kaithi") -> NO_MATCH abstention, zero cards
#    - "no, the original one" -> GS-08 backtrack refines within the set
#    - open the trace view -> validated tool calls, tokens, amortized GPU cost

# 4. STOP — on camera if recording:
make gpu-stop                               # destroys the tagged instance
make gpu-nuke                               # verify: 0 stray instances
```

| Timing | Recorded evidence | Rehearsal window (2026-07-11, instance 442900) |
|---|---|---|
| create → serve window UP (vLLM **+ embed/rerank sidecar**, health-gated) | ~5.5 min LLM-only (P0 validation, 2026-07-01) | **545 s (~9.1 min)** cold — the sidecar adds ~3.5 min over the P0 LLM-only number |
| exports → app live → first cited answer in the UI | n/a (new path) | **40 s** (compose `--wait` + status probe + the first 5.4 s live turn) |
| live turn latency (4 turns through the UI) | P5 window: p50 4535 / p95 5395 ms | **p50 4252 / p95 5211 ms** — parity with the P5 capture |
| full window cost (create → demo → destroy) | $0.34 for the P0 create→smoke→destroy cycle | **832 s total ≈ $0.21** at $0.89/h (+ $0.0028 amortized token cost, 41.4k tokens / 4 turns) |
| warm compile cache | ~100 s (P0 validation) | not exercised — ephemeral create (never a warm box) is the standard path |
| teardown | — | `nuke` → instance 442900 destroyed, **0 stray verified**; the still-running app degraded to the offline state automatically (the failure-mode story, observed live) |

Failure mode: if the GPU dies mid-demo, the app **degrades to Path A automatically**
(structured offline payload + replay browser, HTTP 200, state not corrupted) — which is
itself the demo of the degradation story. *Observed live in the 2026-07-11 rehearsal:
after `gpu-stop`, the still-running app flipped to the offline state on the next status
poll, no restart needed.*

### Demo video (Q2/DEC-P6-3 — RECORDED 2026-07-11)

**Published:** [`p6-demo-v1` Release asset](https://github.com/venkathub/sutradhar-film-finder/releases/download/p6-demo-v1/sutradhar-demo.webm)
(84 s, one-take Playwright screen capture via the committed recorder
`ui/app/e2e/record_demo.mjs`; `DEMO_VIDEO_URL` set as the repo Actions variable + `.env`,
so the offline payload and the static site link it automatically). The take: zero-GPU
replay story → live GPU turns → **the GPU stopped on camera** (the UI degrading to
"offline by design" on the next status poll — the teardown is the closing shot).

To re-record (e.g. with narration over the same visuals): bring up the off-mode server
(`E2E_MODE=off E2E_PORT=8766 uv run python tests/e2e/e2e_server.py`) + a live window
(Path B), run `node e2e/record_demo.mjs` from `ui/app/`, then upload the new asset and
update `DEMO_VIDEO_URL`. Suggested narration beats:

1. **Zero-GPU story (~90 s):** `make demo-up` → offline notice ("offline by design") →
   replay browser → GS-08a: cards, citations, trace view, recorded latencies.
2. **Live story (~2 min):** `make gpu-serve` (cut the ~9 min wait) → exports →
   `make demo-up` → *"which movie is Papanasam a remake of?"* → all five versions,
   original flagged → click a Wikidata citation → *"Kaithi"* decoy → honest no-match →
   *"no, the original one"* backtrack → open the trace view (validated calls, tokens,
   amortized cost).
3. **STOP on camera (~20 s):** `make gpu-stop` → `make gpu-nuke` → *"no stray sutradhar
   instances found"*. The cost story, ended the way it always ends.

## Path C — Rebuild from scratch (the "volume deleted" posture)

The GPU volume is deleted after every window; the whole stack rebuilds from the repo:

```bash
git clone https://github.com/venkathub/sutradhar-film-finder && cd sutradhar-film-finder
cp .env.example .env      # zero secrets needed for the zero-GPU path
make demo-up              # image build -> migrate -> seed-graph-ci -> UI
```

Artifact provenance (what makes this possible):

- **Graph**: seeded from committed recorded fixtures (`data-pipeline/seed_graph_ci.py`)
  — no network, no API keys; the authoritative refresh is `make ingest-seed` (needs
  `TMDB_API_KEY`).
- **Pinned runs** (committed): retrieval `evals/retrieval_runs/` (Table 1 winner),
  generation `evals/generation_runs/` (Table 2 base column + the replay transcripts),
  serving `evals/serving_runs/` (injection/latency evidence), prompt bundles
  `evals/prompts/` (v1.0 `78215ccc…`, v1.1 serving `98b3ece1…`).
- **Hub artifacts**: QLoRA adapter `venkat2393/sutradhar-gemma4-e4b-qlora-v1` (verdict
  CUT — kept for provenance) and the private FT dataset repo (DEC-P4-7).
- **Tool schema**: `docs/phases/tool_schema.v0.json`, frozen, sha256 `4c10ea97…`.

CI proves this path on every push (`demo-smoke`: fresh checkout → `make demo-up` →
health + UI + replay smokes → down).

## Cost table (DEC-0003 discipline)

| Item | Rate / recorded cost |
|---|---|
| A100-PCIE-40GB (JarvisLabs, per-minute billing) | **$0.89/h** (₹84.24/h recorded 2026-07-01) |
| Value alternative (RTX 6000 Ada 48GB, teacher plan-B) | ~$0.99/h (never needed) |
| P0 GPU validation cycle (create → smoke → destroy) | **$0.34 measured** |
| One live interview demo (Path B) | est. $0.25–0.40; **rehearsal measured $0.21** (13.9 min window, 2026-07-11) |
| Standing inference infrastructure | **$0.00 — none exists.** Static surface: GitHub Pages ($0). Langfuse VPS: **deleted 2026-07-11** (DEC-P3-7 amendment 3 — evidence committed, tracing no-ops when unset; `make langfuse-up` rebuilds from scratch on demand) → **standing cost $0.00/month, project-wide** |

**The posture, in one line:** the GPU exists for minutes at a time — benchmark capture
and live demos — and every window ends `destroy`-verified (`make gpu-nuke` → 0
instances). The standing portfolio evidence (benchmarks, pinned transcripts, MLflow
runs, exported traces, the recorded video, this runbook) carries the proof while
everything inference-side is off.
