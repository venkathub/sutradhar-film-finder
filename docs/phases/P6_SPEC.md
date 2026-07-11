# P6 Spec — UI, Containerization, Always-Available Surface & Runbook

> **Status: EXECUTED (2026-07-11).** Tasks 1–12 landed in order on `feature/p6-ui-packaging`;
> rehearsal window measured (545 s cold bring-up, 25 s warm `demo-up`, $0.21/window, teardown
> `nuke`-verified); evidence in `docs/RUNBOOK.md`, BENCHMARKS §"Graceful degradation", and
> `docs/evidence/p6/`. **One human step pending:** the narrated demo video (recording script in
> the RUNBOOK) → Release asset → `DEMO_VIDEO_URL`; no surface renders a video link until then.
>
> **Approved 2026-07-10.** Grooming complete: D1–D5 and Q1–Q5 confirmed by the user
> with the recommended options (§3/§7); logged as **DEC-P6-1..5** in `docs/DECISIONS.md`.
> Web-research pass done 2026-07-10 (§8) — toolchain versions, hosting limits, and attribution
> obligations below are sourced, not assumed. Implementation follows the §5 task order.
>
> P6 packages everything the previous phases proved into the product and portfolio surface:
> the chat UI (versions with the original flagged, per-claim citations, trace view, graceful
> degradation), full-stack containerization with a one-command bring-up, the static
> always-available portfolio surface, and `docs/RUNBOOK.md` with a rehearsed, **timed**
> on-demand GPU demo path. ROADMAP P6 exit criteria are the contract; risks R4 (demo bring-up
> speed) and R5 (evidence must carry the portfolio without a live endpoint) are discharged here.
>
> **Entry condition (met):** P5 EXECUTED (2026-07-05, `servewin-25c029d3` sealed; DEC-P5-1..6).
> **Standing FT verdict: CUT (DEC-P4-9)** — the UI demos the well-prompted base
> `google/gemma-4-E4B-it` under the v1.1 serving prompt bundle; if P4.1 ever flips to KEEP,
> that is an env/config change invisible to everything P6 builds. P6 does not wait on P4.1.

---

## 1. Scope

### In scope

1. **Chat UI (`/ui`)** — the find-a-movie chat against the existing P5 API:
   - conversation panel over `POST /api/chat` (multi-turn; `conversation_id` carried so GS-08
     backtracking works through the same DEC-P5-2 session store);
   - **version-set rendering**: every language version as a card with the **original flagged**,
     the typed relationship badge (`is_original_of` / `is_remake_of` / `is_official_dub_of` /
     `is_unofficial_remake_of` / `is_sequel_of`), `relationship: null` rendered honestly as
     "unverified relationship" (never a guessed label), year/language/lead cast, and the
     `confidence` tier badge (HIGH/MEDIUM);
   - **per-claim citations**: `sources[]` rendered as clickable provenance — Wikidata QID →
     entity URL, TMDB id → movie URL, IMDb tt-id → title URL, Wikipedia → page URL **pinned to
     the stored revision id** (LICENSING.md obligation), `rule` → tooltip naming the
     deterministic rule, `human` → "human-verified" gate note;
   - **trace view**: how the answer was assembled — per-turn tool calls (name + arguments +
     result summary + latency), schema-validation outcomes, guardrail `warnings[]`, usage/cost;
     tool labels **generated from `tool_schema.v0.json`**, never hand-written (the DEC-P1-8
     posture, extended to the UI);
   - **graceful-degradation state**: GPU off (the default) renders the structured offline
     payload — "live demo offline by design" + benchmark links + the recorded **demo video** +
     a **replay browser** over `GET /api/replay/{fixture}` so the Papanasam story plays with
     zero GPU;
   - **attribution chrome** (the LICENSING.md P6 obligations, per-source terms verified in the
     §8 research pass):
     - **TMDB** — the exact FAQ requirements: the TMDB logo identifies the API use, the notice
       *"This product uses the TMDB API but is not endorsed or certified by TMDB."* is placed
       prominently, and the TMDB logo is rendered **less prominent than Sutradhar's own mark**
       (an explicit TMDB condition, previously missing from our docs — LICENSING.md row updated
       at execution);
     - **Wikipedia** — per-claim article link satisfies attribution (WP:Reusing Wikipedia
       content: "a link back to the article is generally thought to satisfy the attribution
       requirement"); we go one better and link the **stored revision** (`?oldid=`), plus a
       visible "CC BY-SA 4.0" license label per the Wikimedia content-reuse guidance
       (title/source/license);
     - **IMDb** — AKA/dub titles in version cards derive from `title.akas`, so the footer carries
       the LICENSING.md courtesy line *"Information courtesy of IMDb (https://www.imdb.com).
       Used with permission."* + the non-commercial-demo note. (Gap found in this grooming pass:
       the draft spec covered TMDB/Wikipedia but not IMDb — the UI is the first surface that
       *displays* IMDb-derived data.)
2. **API additions the UI needs (no tool-schema change, additive only):**
   - `ChatResponse.trace: list[TraceStep]` — the structured per-call record the orchestrator
     already produces internally (D4);
   - `GET /api/replays` — promote the existing `available_replays()` (currently only in the 404
     body) to a first-class discovery route;
   - `DEMO_VIDEO_URL` setting wired into the offline payload (discharges the P5
     `"demo_video": null` placeholder);
   - FastAPI serves the built UI as static files (same-origin → no CORS surface).
3. **Full-stack containerization**: an `app` image (API + built UI assets) joining the existing
   compose stack (Postgres+pgvector, Redis); **one-command bring-up** (`make demo-up`) that
   works on a fresh clone with zero GPU (degradation + replay experience) and flips to the live
   experience by exporting the `gpu-serve` endpoints; a CI job proves the fresh-clone path.
4. **Static always-available surface** (`site/`): landing page + architecture diagram +
   **benchmark report generated from `docs/BENCHMARKS.md`** + recorded demo video + links to the
   standing evidence (MLflow screenshots, exported Langfuse traces, RUNBOOK). Hosting per D3.
   **Never serves any neural model** — static and precomputed content only (CLAUDE.md).
5. **`docs/RUNBOOK.md`**: the rehearsed live-demo path (JarvisLabs bring-up → `make gpu-serve`
   → export env → `make demo-up` → demo in the UI → **STOP**, teardown `nuke`-verified), the
   zero-GPU demo path, rebuild-from-scratch steps (fresh clone → seeded graph → pinned
   artifacts), cost table per DEC-0003, and the **measured bring-up time** from the rehearsal.
6. **One short GPU rehearsal window (~$0.5–1, inside the DEC-0003 envelope):** time the
   bring-up (R4 evidence), run the live UI demo, **record the demo video**, capture UI + dashboard
   screenshots; teardown guaranteed.
7. **Portfolio packaging**: top-level README ties the whole system together;
   `docs/PORTFOLIO.md` finalized with quantified bullets; BENCHMARKS evidence sections updated
   (see §6 — Tables 1 and 2 numbers are **not** touched).

### Non-goals (explicit — prevents scope creep)

- **No retrieval or model changes.** Chunking/fusion/θ/prompts/verdict all stay pinned; **no
  pinned metric artifact changes**; Tables 1 & 2 are never re-scored by this phase.
- **No P4.1 work** — it remains a separate, budget-gated phase; the UI is model-agnostic via env.
- **No tool-schema change** — the UI consumes `/api/chat`; v0 is consumed unchanged (a
  wording-only status note at exit, as in P2/P3/P4/P5).
- **No new API capabilities beyond §1.2** — no auth, multi-tenancy, rate-limiting products, no
  user accounts, no persistence of conversations beyond the existing TTL store.
- **No token streaming (SSE/WebSocket)** unless D2 decides otherwise — recommended cut.
- **No 24/7 inference deployment, no public unattended LLM endpoint.** The deployed UI's live
  path only functions during a GPU window; the standing deployment is the static surface +
  (optionally) the degradation-mode app. This is the cost-discipline feature, not a gap.
- **No new observability stack** (Langfuse + MLflow settled, DEC-P3-2/6/7); no standing
  Prometheus/Grafana (DEC-P5-6).
- **No i18n of UI chrome** (queries themselves are multilingual; labels stay English), no
  mobile-native app, no dark/light theming beyond a single clean theme, no CMS on the static
  surface.
- **No Java** — DEC-P5-1's cut stands; nothing in P6 reopens it (§2.7).

---

## 2. Design

### 2.0 Gating-story traceability (what P6 makes true)

| ROADMAP clause | P6 component | Evidence artifact |
|---|---|---|
| *"returns all language versions … original flagged"* (Table A) | version-set cards | UI E2E `test_ui_version_set_recall_gs01/gs06`; screenshots |
| *"clicks a citation and sees the exact Wikidata/TMDB record and confidence"* (Table A) | citation rendering | citation href/contract tests; screenshot |
| *"offers a trace view of how it reasoned"* (Table A) | trace view over `ChatResponse.trace` / replay transcripts | E2E + v0-validation test |
| *"when the GPU is off, the same story replays from recorded evidence"* (Table B) | degradation state + replay browser + demo video | zero-GPU E2E; `make demo-up` |
| *"GPU brought up on-demand … then stopped"* (Table B, R4) | RUNBOOK + rehearsal window | measured bring-up time; teardown log |
| Standing evidence artifacts (Table B, R5) | static surface + README/PORTFOLIO | published site; benchmark report page |

### 2.1 Component breakdown

| Component | Location | New/Change | Role |
|---|---|---|---|
| Frontend app | `ui/app/` (per D1) | **new** | Chat, version cards, citations, trace view, offline/replay states. Frontend assets only — no `sutradhar.*` package (per the P0 stub note) |
| Tool-label map generator | `ui/` build step | **new** | Emits the UI's tool name/param labels **from `docs/phases/tool_schema.v0.json`**; a drift test fails if the artifact and the generated map diverge |
| Trace payload | `sutradhar.serving.schemas` + `orchestrator` | **change (additive)** | `TraceStep` model; orchestrator assembles it from the per-call records it already validates/traces (D4) |
| Replay discovery | `sutradhar.serving.degrade` + `app` | **change (additive)** | `GET /api/replays`; replay payload shaped so live and replayed turns render through the same UI components |
| Static file serving | `sutradhar.serving.app` | **change** | Mounts the built UI (`ui/app/dist/`) at `/`; API stays under `/api/*` (same-origin) |
| App container | `infra/` Dockerfile + compose | **new** | Multi-stage build (UI build stage → `uv` runtime stage); `app` service joins compose; `make demo-up` |
| Static surface | `site/` + generator in `sutradhar` or scripts | **new** | Landing page, architecture diagram, benchmark report generated from `BENCHMARKS.md`, video + evidence links; deployed per D3 |
| Runbook | `docs/RUNBOOK.md` | **new** | Live-demo, zero-GPU, and rebuild-from-scratch paths with measured timings + costs |
| Rehearsal capture | `infra/gpu/jarvis.py` `serve` session (reused) | **reuse** | The P5 `serve` session is the building block; P6 adds nothing to it — the RUNBOOK wraps it |

**Reused unchanged:** the whole P5 request path (orchestrator, guardrails, sessions, degrade,
cost accounting), `LLMClient`, tracing seam, pinned prompt bundles, compose Postgres/Redis,
`make gpu-serve`/`gpu-stop`/`gpu-nuke`.

### 2.2 Data models & contracts

**`TraceStep`** (pydantic, `extra="forbid"`), appended to `ChatResponse` as `trace: list[TraceStep] = []`
— an additive, backward-compatible extension of the §2.2 P5 contract (existing consumers ignore it):

```jsonc
{ "step": 1,
  "tool": "get_versions",                       // v0 tool name — validated, never free text
  "arguments": { "work_id": "…", "scope": "indian", "include_sequels": true },
  "valid": true,                                // validate_emitted_call outcome
  "validation_error": null,                     // populated when valid=false (fed-back call)
  "result_summary": { "kind": "versions", "count": 5, "ids": ["…"] },  // summary, not the blob
  "latency_ms": 210.4 }
```

Design rules carried over: the trace shows **what the orchestrator already validated** — the UI
never re-derives or invents tool semantics; `result_summary` is bounded (no full tool-result blobs
in the response; the versions/citations fields already carry the user-facing content).

**Replay contract:** `GET /api/replay/{fixture}` (existing) already returns messages + tool calls +
answers + latencies from the pinned generation run; P6 adds a thin adapter so a replayed turn and a
live turn render through the same components (one rendering path = one set of tests).

**Citation link builders** (pure functions, unit-tested per `SourceId` variant):
`wikidata → https://www.wikidata.org/wiki/{ref}`; `tmdb → themoviedb.org/movie/{ref}`;
`imdb → imdb.com/title/{ref}`; `wikipedia → page URL + ?oldid={revision}` (revision-pinned per
LICENSING.md); `rule` → no link, named-rule tooltip; `human` → gate note. Confidence badge from the
row's `confidence`.

**New/changed env (`.env.example` + `Settings`):** `DEMO_VIDEO_URL` (default empty → omitted from
the offline payload), `SITE_BASE_URL` (static-surface canonical URL, docs/link-check only). No new
secrets.

### 2.3 Request/data flow

```
Browser (static UI, same origin)
 ├─ GET /            → built UI assets (FastAPI StaticFiles / container)
 ├─ GET /api/status  → cached degradation state (30 s TTL, DEC-P5-5) → UI picks live vs offline mode
 ├─ live mode:  POST /api/chat {conversation_id, message}
 │     → ChatResponse {answer, versions[], citations[], warnings[], usage, trace[]}
 │     → UI renders answer + version cards + citations; trace view from trace[]
 │     → turn 2 "no, the original one" sends the same conversation_id (GS-08 path)
 ├─ offline mode: offline payload {detail, evidence{benchmarks, replay, demo_video}}
 │     → GET /api/replays → list → GET /api/replay/{fixture} → same rendering path
 └─ footer: TMDB logo/disclaimer + Wikipedia CC BY-SA + links to static surface
```

The static surface is fully decoupled: built from committed evidence at deploy time, zero runtime
dependency on the app, DB, VPS, or GPU.

### 2.4 Containerization & bring-up

- **`app` image**: multi-stage — stage 1 builds the UI (**`node:24` LTS builder** — build-time
  only, not a neural op; ROADMAP §2 compute placement is untouched); stage 2 follows the
  **official Astral uv-in-Docker multi-stage pattern** (§8: builder runs `uv sync --frozen
  --no-dev` with cache mounts; the final image copies only the venv + app — no uv, no compilers,
  the same `uv.lock` CI uses so image and laptop cannot drift). Runs
  `uvicorn sutradhar.serving.app`. Model endpoints stay env-driven and **empty by default**
  (off = first-class state).
- **`make demo-up`** = compose up (postgres, redis, app) → migrate → seed graph from recorded
  fixtures (`seed-graph-ci` path) → open the UI. Fresh clone → working degradation + replay
  experience in one command, zero GPU, zero secrets. This is the 30-second demo path.
  (The infra README's snap-Docker `$HOME` staging workaround applies unchanged and is restated
  in the RUNBOOK.)
- **Live flip**: `make gpu-serve` (P5 session, unchanged) prints `LLM_BASE_URL`/`EMBED_BASE_URL`/
  `RERANK_BASE_URL`; `make demo-up` picks them from the environment — no rebuild.
- CI: a Tier-1 job builds the image and smokes `/api/health`, `/` (UI index), and one replay from
  a fresh checkout.

### 2.5 Static always-available surface

- `site/` generator renders: landing (the gating story in one screen), architecture diagram
  (committed SVG/PNG under `docs/assets/`), **benchmark report page generated from
  `BENCHMARKS.md`** (single source of truth — no hand-copied numbers), demo video embed/link,
  evidence links (committed trace exports, MLflow screenshots, RUNBOOK, repo).
- Deployed per **D3** (recommendation: GitHub Pages via the official `actions/deploy-pages` flow
  on merge to main). The DEC-P3-7 VPS keeps serving Langfuse only; the "intended P6 consolidation
  host" clause is resolved by D3. The demo **video is a GitHub Release asset, not a site file**
  (D3 sizing note) — the site stays far under the 1 GB Pages cap.
- Link-check + required-assets test gate the build (a dead evidence link is a CI failure, not a
  surprise during an interview).

### 2.6 RUNBOOK & the rehearsal window

`docs/RUNBOOK.md` documents three paths, each rehearsed:
1. **Zero-GPU demo (default)** — `make demo-up` → UI replay browser → talk over the recorded
   benchmark. Target: < 1 min from a warm laptop.
2. **Live interview demo (R4)** — pre-call: JarvisLabs resume/create + `make gpu-serve` →
   health-gated endpoints → export env → `make demo-up` → live Papanasam turn + GS-08 backtrack →
   **STOP** (`make gpu-stop`, `gpu-nuke` check). The rehearsal window measures and records
   create→ready and resume→ready times; the measured numbers go in the RUNBOOK and the
   BENCHMARKS degradation section.
3. **Rebuild from scratch** — fresh clone → `.env` from example → `make demo-up`; artifact
   provenance (pinned runs, HF Hub) restated. Proves the "volume deleted, stack rebuilt" posture.

The same window records the **demo video** (script: Tanglish query → full version set with
original flagged → citation click → GS-02 decoy → NO_MATCH → "no, the newer one" backtrack →
trace view → GPU stopped on camera) and captures UI/dashboard screenshots.

### 2.7 Python vs (optional) Java

Settled — not reopened. DEC-P5-1 cut the Java/Spring gateway; P6 adds no server-side surface that
could reopen it. The UI itself is TypeScript/JS **frontend assets** (or Jinja templates under
D1-B) — there is no JVM anywhere in the stack. Python continues to own the API; the only new
runtime is the node *build-time* toolchain (never deployed, never serving).

### 2.8 Tool-schema conformance statement

P6 **consumes v0 unchanged — no new tool, no signature change, no version bump.** The UI never
calls tools; it renders tool calls the orchestrator already validated. Conformance is still
enforced in both directions: (a) `TraceStep.tool`/`arguments` for every step in every committed
replay transcript and trace fixture must validate against `tool_schema.v0.json` (reusing
`sutradhar.toolcalls.validate_emitted_call` — no hallucinated tool or parameter names can render);
(b) the UI's tool-label map is **generated from the JSON artifact** with a drift test. At exit,
`TOOL_SCHEMA.md` gains the customary wording-only status note ("v0 renders the P6 trace view
unchanged"), noted for DECISIONS — not a schema version event.

---

## 3. Decisions — CONFIRMED 2026-07-10 (logged as DEC-P6-1..5 in docs/DECISIONS.md)

Settled decisions were **not** reopened: FastAPI-only gateway (DEC-P5-1), Redis sessions
(DEC-P5-2), injection defense + prompt v1.1 (DEC-P5-3), sidecar serving topology (DEC-P5-4),
caching scope (DEC-P5-5), dashboards (DEC-P5-6), Langfuse VPS (DEC-P3-7), model stack/verdict
(DEC-0001/P4-9), vector store/retrieval config (DEC-P2-1..6). The five P6 choices below were
confirmed with their recommended options; mapping: D1 (+ Q3) → DEC-P6-1, D2 → DEC-P6-2,
D3 (+ Q1/Q2/Q4/Q5) → DEC-P6-3, D4 → DEC-P6-4, D5 → DEC-P6-5.

### D1 — UI stack

| Option | Trade-offs |
|---|---|
| **A. Vite + React + TypeScript SPA, built to static assets (recommended)** | The "product-quality UI" skill signal (CLAUDE.md audience); componentized version cards/trace view; vitest + Playwright are first-class; matches the P0 stub's "frontend assets" framing. Cost: a node build toolchain on the laptop/CI (build-time only, cheap) and a second language in the repo |
| B. Server-rendered Jinja2 + htmx | One runtime, zero node, all tests stay in pytest; but interactive trace view/citation popovers get awkward, and the portfolio reads "backend engineer avoided frontend" — the opposite of the P6 skill row |
| C. Streamlit/Gradio | Fastest to demo; weakest product signal, poor control over citations/trace layout, heavyweight runtime dependency for a static-servable UI |

**Recommendation: A.** The UI is a named skill artifact ("Product UI with citations + trace
view", ROADMAP §4); A is the only option that produces pure static assets the same container and
the static host can both serve. React over Preact/Svelte purely for reviewer familiarity.
**Toolchain pins (verified current, §8):** **Node 24 LTS "Krypton"** (Active LTS; Node 22 is in
maintenance — new toolchain starts on the active line), **Vite 8.x** (stable since 2026-03-12 on
the unified Rolldown bundler; 8.1 current as of 2026-06-23 — no reason to start a new app on the
legacy Rollup majors), **React 19**, **Vitest 4** — exact versions locked in `ui/app/package.json`
+ lockfile committed (the `uv.lock` discipline, applied to node).

### D2 — Response delivery: keep non-streaming JSON vs add SSE

| Option | Trade-offs |
|---|---|
| **A. Non-streaming JSON + deterministic progress states in the UI (recommended)** | Zero server change; the P5 contract stays pinned; measured p50 4.5 s / p95 5.4 s per turn is comfortably covered by a staged "parsing → searching the graph → composing" indicator. One rendering path for live + replay |
| B. SSE phase events (tool-call started/finished), final JSON unchanged | Honest progress from real orchestrator events; but a new endpoint + orchestrator callback surface + a second client path to test, for a demo whose turns are ~5 s |
| C. Full token streaming | Best perceived latency; but the agent loop interleaves tool rounds (tokens arrive late anyway), the output gate (no-hallucinated-movie) must see the **complete** answer before display — streaming would show then retract an invention, breaking the guardrail story |

**Recommendation: A.** C is actively wrong for the output-gate semantics; B is deferred as the
documented upgrade. This finalizes the "SSE is P6 polish if wanted" clause from DEC-P5-1 as a cut.

### D3 — Static-surface hosting

| Option | Trade-offs |
|---|---|
| **A. GitHub Pages, deployed from `site/` by CI (recommended)** | $0, zero standing service, survives even VPS deletion (maximal "always-available"), no tunnel dependency, evidence lives beside the repo an interviewer is already reading. Limits comfortably fit (verified, §8): 1 GB published site, 100 GB/mo soft bandwidth, 10 builds/hr; the repo is already public (P0 branch-protection record) and the site is a non-commercial portfolio — inside the Pages acceptable-use terms. Cost: github.io URL unless a custom domain is added |
| B. Consolidate on the DEC-P3-7 AIC VPS (Caddy static + cloudflared named tunnel on a real domain) | One self-hosted home for Langfuse + site (the "intended consolidation host" wording); but the VPS has **no inbound 443** (P3 finding) so the *always-available* surface would depend on an outbound tunnel + a ₹799/mo box — a single point of failure for the one artifact that must never be down. A named tunnel additionally requires owning a domain with its DNS delegated to a (free) Cloudflare zone (§8) — a new external dependency A avoids entirely |
| C. Cloudflare Pages / Netlify | Also free/robust; adds a third-party account for no capability GitHub Pages lacks here |

**Recommendation: A.** The VPS stays Langfuse-only (its evidence is exported + committed anyway);
this *resolves* DEC-P3-7's "decided in P6" clause rather than reopening it. The optional
read-only MLflow mirror on the VPS is recommended **cut** (committed screenshots + run links
already serve the evidence; a public MLflow adds ops/attack surface for zero new proof) — §7 Q4.
**Demo-video placement (research-grounded):** the video does NOT go in the Pages site or git
history — a **GitHub Release asset** carries it (per-file limit 2 GiB, **no bandwidth or
total-size limit** on release assets, vs Pages' 1 GB whole-site cap); the site and the offline
payload link it via `DEMO_VIDEO_URL`.

### D4 — Trace-view data source

| Option | Trade-offs |
|---|---|
| **A. Additive `trace[]` on `ChatResponse`, assembled in-process by the orchestrator (recommended)** | The data already exists at the validation seam; deterministic, testable with the scripted fake client; replay transcripts adapt to the same shape — one rendering path; no secrets |
| B. UI queries the Langfuse API | Duplicates what the server already knows; couples the demo UI to the VPS and would put observability credentials in a browser — disqualifying |
| C. Persist tool records in Redis, new `GET /api/trace/{conversation_id}` | Keeps ChatResponse small; but grows session state + TTL semantics for data the client wanted at turn time anyway |

**Recommendation: A.** Bounded `result_summary` keeps the payload small; Langfuse remains the
ops view (trace_id already links them).

### D5 — App containerization shape

| Option | Trade-offs |
|---|---|
| **A. Single `app` image: multi-stage UI-build → uv runtime; FastAPI serves the static UI (recommended)** | One image, one service added to compose, same-origin (no CORS), fastest `make demo-up`, cleanest rebuild-from-scratch |
| B. Separate nginx container for UI + API container | "Production-shaped" static serving; but +1 container/CI leg and a reverse-proxy config for a demo stack with no TLS/scale need — the DEC-P5-1 rationale applies |
| C. No app container (host uvicorn + `ui-dev` server) | Least work; fails the "full stack containerized, one-command bring-up" exit criterion outright |

**Recommendation: A.** B's split is the documented future-ops note if the app ever fronts real
traffic.

---

## 4. Test strategy

P6 touches neither retrieval nor generation quality, so **no eval thresholds change and no
pinned metric artifact may change** — that invariant is itself asserted (existing Tier-1 pinned-
artifact regression tests must pass untouched; the reproducibility stamps P6 cites are the P5/P4
ones). What P6 must prove is that the *rendering, packaging, and demo paths* preserve the gated
behaviour. All app/UI tests are Tier-1 (no GPU, no model — scripted `LLMClient` fake + live DB /
fakeredis, the proven P5 integration pattern).

### Unit (pytest, Tier-1)
- `TraceStep` assembly: validated calls, invalid-call feedback steps (`valid=false` +
  `validation_error`), bounded `result_summary`, latencies present.
- Citation link builders: one test per `SourceId` variant (wikidata/tmdb/imdb/wikipedia+oldid/
  rule/human); Wikipedia links MUST carry the pinned revision id (the WP:Reusing-content
  link-back attribution, made revision-exact).
- **Attribution compliance test** (`test_ui_attribution_obligations`): the rendered chrome
  carries the exact TMDB notice string, the TMDB logo asset, the IMDb courtesy line, and the
  CC BY-SA 4.0 label — a LICENSING.md obligation as an executable check, not a checklist item.
- `GET /api/replays` route; replay→UI-shape adapter (a replayed GS-08 turn produces the same
  render model as a live turn).
- Offline payload: `DEMO_VIDEO_URL` set/unset; still HTTP 200, never 5xx.
- Static-surface generator: benchmark page numbers come from `BENCHMARKS.md` (no literals),
  required assets present (diagram, video URL, evidence links), link check green.

### UI component tests (Vitest 4 Browser Mode — Playwright provider — + Testing Library, Tier-1)
Component tests run in a **real browser via Vitest 4's stable Browser Mode** (Playwright
provider), not jsdom — the current recommended practice (§8) and it reuses the same Playwright
install the E2E tier needs (one browser dependency, headless in CI):
- Version card: original flag rendering; each of the five edge labels; `relationship: null` →
  "unverified relationship"; confidence badges.
- NO_MATCH answer renders the abstention state and **zero version cards**.
- `abstain=true`-with-results renders as "low confidence", never as fabricated certainty
  (the DEC-P2-5 honesty semantics, now visual).
- Trace view: renders only v0 tool names (generated label map); unknown tool name in a fixture →
  explicit error state, not a silent render.
- Offline state: evidence links + replay browser; attribution footer (TMDB notice + logo
  less-prominent rule, Wikipedia CC BY-SA + revision links, IMDb courtesy line) present on every
  page.

### End-to-end (Playwright against the containerized app, scripted fake LLM, Tier-1)
The **named golden regressions**, asserted on the rendered DOM (fixtures drive the fake client
with the pinned expected tool flows — same fixtures that gate P2/P3):
- `test_ui_version_set_recall_gs01` / `test_ui_version_set_recall_gs06` — all five Drishyam
  versions (GS-01) / the full sequel+remake franchise (GS-06) rendered, original flagged:
  rendered version-set recall = 1.0.
- `test_ui_no_hallucinated_movie_gs02` — decoy query renders NO_MATCH; no movie card, no
  invented title anywhere in the DOM (detector re-run over rendered text).
- `test_ui_dub_vs_remake_gs04` — Baahubali versions all badge `is_official_dub_of`; the string
  "remake" appears on no card.
- `test_ui_sibling_vs_remake_gs05` — Devdas adaptations render as siblings linked to the novel
  (`based_on`), never as a remake chain.
- `test_ui_false_merge_gs10` — "Vikram Kamal Haasan" renders two distinct works (or the
  disambiguation ask); never one merged card set.
- `test_ui_backtracking_gs08` — three turns over HTTP through the real session store; turn 2
  correction updates the highlighted version without losing the set.
- `test_ui_gpu_off_replay` — status=off → offline state → replay GS-08a renders the full
  transcript including the trace view.

### Tool-schema conformance (the tool-calling test this phase requires)
- `test_ui_trace_tool_calls_validate` — every tool call in every committed replay transcript,
  every `TraceStep` in the E2E fixtures, and the pinned generation-run transcripts the replay
  browser exposes validates against `tool_schema.v0.json` via `validate_emitted_call` — **no
  hallucinated tool or parameter name can reach a rendered trace**.
- `test_ui_tool_labels_generated` — the UI label map is byte-derived from the v0 JSON artifact
  (drift fails CI).

### Packaging & operations
- CI fresh-checkout job: build the `app` image → `make demo-up` → smoke `/api/health`, `/`,
  one replay → down. (The "runs cleanly from scratch" DoD line, automated.)
- Static-surface build + link-check job; deploy on merge to main (D3).
- Tier-2 (the one GPU rehearsal window): live UI turn parity spot-check (same GS-01 flow through
  the real model — a smoke, not a re-benchmark), timed bring-up capture, video/screenshot
  capture; teardown `nuke`-verified. No new metrics gates — evidence capture only.

---

## 5. Task breakdown (ordered, independently committable)

1. **API: trace + replay discovery** — `TraceStep`, `ChatResponse.trace`, orchestrator assembly,
   `GET /api/replays`, `DEMO_VIDEO_URL` in the offline payload; unit tests. (Additive; no UI yet.)
2. **UI scaffold (D1)** — `ui/app/` toolchain, `make ui-build`/`ui-dev`, FastAPI static mount,
   generated tool-label map + drift test, CI node job.
3. **Chat panel + degradation states** — status polling, live/offline modes, replay browser over
   `/api/replays`; component tests.
4. **Version-set rendering** — cards, original flag, relationship badges (incl. null →
   unverified), confidence; component tests.
5. **Citations + attribution chrome** — link builders, per-claim popovers, TMDB logo/disclaimer,
   Wikipedia CC BY-SA footer; tests.
6. **Trace view** — live `trace[]` + replayed transcripts through one rendering path;
   `test_ui_trace_tool_calls_validate`.
7. **Playwright E2E suite** — the seven named tests in §4 against the fake-client app.
8. **Containerization (D5)** — multi-stage `app` image, compose service, `make demo-up`,
   fresh-checkout CI smoke job.
9. **Static surface (D3)** — `site/` generator (benchmark page from BENCHMARKS.md), diagram,
   link-check, CI deploy.
10. **`docs/RUNBOOK.md`** — three rehearsed paths written (timings marked "measured in task 11").
11. **GPU rehearsal window (~$0.5–1)** — timed bring-up, live UI smoke, **demo video recorded**,
    screenshots; teardown verified; `DEMO_VIDEO_URL` set; evidence committed.
12. **Docs close-out** — BENCHMARKS degradation/serving evidence updates (video, screenshots,
    measured bring-up time, site URL), TOOL_SCHEMA wording-only note, DEC-P6-1..5 logged,
    **LICENSING.md updated** (TMDB logo-prominence clause + IMDb UI courtesy line + site/video
    distribution rows), top-level README + module READMEs + PORTFOLIO.md finalized.

---

## 6. Definition of Done (instantiated from CLAUDE.md)

- [ ] Code complete and matching this approved spec (D1–D5 as confirmed).
- [ ] Unit + component + E2E + fresh-checkout container tests written and passing in Tier-1 CI;
      ruff/mypy-strict clean.
- [ ] **Eval thresholds:** no retrieval/generation change in this phase — the existing Tier-1
      gates (pinned Table 1 run, pinned generation run, injection dry-run, golden regressions)
      pass **untouched**; asserted, not assumed. The Tier-2 rehearsal window records evidence
      only (no new metric gates).
- [ ] **Benchmark tables:** Tables 1 and 2 are **NOT updated** — P6 changes no model or
      retrieval behaviour (the two-table honesty rule). The **"Serving & guardrails / Graceful
      degradation" evidence sections ARE updated**: demo-video link, UI screenshots, measured
      GPU bring-up time (R4), static-surface URL, rehearsal-window reproducibility stamp
      (code SHA · prompt bundle v1.1 hash · v0 sha256 · pinned runs · GPU SKU/$).
- [ ] `ui/README.md` (purpose/architecture/run/tests), `serving/README.md` + `infra/README.md`
      updates, `docs/DECISIONS.md` DEC-P6-1..5, TOOL_SCHEMA.md status note.
- [ ] Runs cleanly from scratch: fresh clone + `.env` from example → `make demo-up` → working
      zero-GPU demo (CI-proven).
- [ ] **30-second demo path:** `make demo-up` → UI opens → replay GS-08a with citations + trace
      view, zero GPU. (The live path is the RUNBOOK's separate, timed flow.)
- [ ] `docs/RUNBOOK.md` complete with measured timings; teardown/rebuild documented;
      **no 24/7 inference deployment exists** — verified by `make gpu-nuke` at window end.
- [ ] `docs/PORTFOLIO.md` finalized with quantified bullets (incl. total GPU spend across the
      project and the "nothing inference-side runs 24/7" cost story).

---

## 7. Open questions — RESOLVED (2026-07-10, user-confirmed with the recommended options)

- **Q1 — Domain → no custom domain.** The static surface lives at the `github.io` URL; Langfuse
  stays on its current tunnel URL. (Logged in DEC-P6-3.)
- **Q2 — Demo video → APPROVED, GitHub Release asset.** One ~3–5 min video recorded in the
  task-11 window covering both stories (zero-GPU replay + live GPU turn + STOP on camera);
  hosted as a Release asset (< 2 GiB/file, no bandwidth/total-size limit — §8), linked via
  `DEMO_VIDEO_URL`, never committed to git history or the Pages site. (Logged in DEC-P6-3.)
- **Q3 — Node toolchain → ACCEPTED.** Node 24 LTS on laptop/CI, build-time only, no deployed
  runtime; D1-A stands. (Logged in DEC-P6-1.)
- **Q4 — MLflow mirror → CUT.** Committed screenshots + run links suffice; no public MLflow on
  the VPS. (Logged in DEC-P6-3.)
- **Q5 — Standing app deployment → static surface only.** The degradation-mode app stays a
  one-command local/live-window artifact — nothing standing that looks like a chat endpoint;
  the no-24/7 posture stays unambiguous. (Logged in DEC-P6-3.)

---

## 8. Sources / research annex (web-research pass, accessed 2026-07-10)

**Attribution & licensing obligations (drives §1.1 attribution chrome + the LICENSING.md update):**
- TMDB API FAQ / Terms (via TMDB support threads quoting the FAQ verbatim) — three concrete
  requirements: use the TMDB logo to identify API use; place *"This product uses the TMDB API
  but is not endorsed or certified by TMDB."* prominently; **the TMDB logo must be less
  prominent than the application's own primary mark**. The prominence clause was absent from
  LICENSING.md — added as a P6 doc update.
- *Wikipedia: Reusing Wikipedia content* + *Wikipedia: Copyrights* — text is CC BY-SA 4.0
  (Wikimedia moved to CC 4.0 in June 2023); "a link back to the article is generally thought to
  satisfy the attribution requirement". Wikimedia *APIs/Content reuse* guidance: attribution =
  title, author/source, license. Our per-claim `?oldid=` revision links (already stored on every
  `plot_texts` row) exceed the bar.
- IMDb non-commercial datasets (LICENSING.md row, unchanged terms) — the UI is the first surface
  displaying IMDb-derived AKA titles → the courtesy line moves from docs into rendered chrome.

**Hosting & evidence distribution (drives D3 + Q2):**
- GitHub Docs *GitHub Pages limits* (+ 2026 tier summaries) — 1 GB published-site cap, 100 GB/mo
  soft bandwidth, 10 builds/hr; free on public repos; prohibited for commercial/SaaS use — a
  non-commercial portfolio surface fits. Deployment via the official Actions Pages flow.
- GitHub Docs *About releases* — ≤ 1,000 assets/release, **each file < 2 GiB, no limit on total
  release size or bandwidth** — the right home for the demo video (vs the 100 MB git-file limit
  and the 1 GB Pages cap).
- Cloudflare Tunnel docs + setup guides — a **named** tunnel (stable hostname) requires a domain
  whose DNS is delegated to a (free) Cloudflare zone; quick tunnels get random URLs. Grounds the
  D3-B dependency accounting and Q1.

**Frontend toolchain (drives D1 + §4 tooling):**
- vite.dev *Announcing Vite 8.0* (2026-03-12) and *Vite 8.1 is out!* (2026-06-23) — Vite 8 is the
  stable line on the unified Rust-based Rolldown bundler (41.6 M weekly downloads by June 2026);
  new apps start on 8.x, not legacy Rollup majors.
- nodejs.org release schedule + Node 24 line (24.11.0 LTS) — **Node 24 "Krypton" is Active LTS**;
  Node 22 is in maintenance; June 2026 security releases patched 22.x/24.x/26.x — pin the active
  LTS and take patch releases.
- Vitest docs *Component Testing / Browser Mode* (+ 2026 guides) — **Vitest 4 Browser Mode is
  stable** and the recommended way to test components in a real browser via the Playwright
  provider (jsdom demoted to a fallback); pairs with Playwright E2E on one browser install.
- React 19 stable (current major; Vite + React 19 is the default pairing in 2026 guides).

**Containerization (drives §2.4 / D5):**
- Astral *Using uv in Docker* + `astral-sh/uv-docker-example` `multistage.Dockerfile` — the
  official multi-stage pattern: builder stage `uv sync --frozen --no-dev` with cache mounts;
  final image copies the venv only (no uv/compilers). Community measurements report large image
  and CI-time reductions; matches the DEC-P0-1 lockfile discipline.

**In-repo precedents (the reuse contract):** P5 `serve` session + degradation payload +
`/api/replay` (the UI's two data paths); `available_replays()`; the P3/P5
generated-tools-array posture extended to the UI tool-label map; DEC-P2-6 committed-artifact CI
gating (the pinned runs the replay browser renders); infra README's cold create→destroy timing
evidence (the RUNBOOK's baseline for the warm-resume measurement); LICENSING.md P6 obligations.
