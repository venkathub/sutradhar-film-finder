# ui

Find-a-movie chat interface with version cards, per-claim citations, and a trace view.
**Frontend assets only — no `sutradhar.*` import package; no neural model runs here.**

## Architecture (P6, DEC-P6-1)

- **Stack:** Vite 8 + React 19 + TypeScript SPA under `ui/app/`, built to pure static
  assets (`ui/app/dist/`) that FastAPI serves same-origin at `/` (API stays under
  `/api/*` — no CORS surface). Node 24 LTS is a **build-time** toolchain only: it is
  never deployed and never serves anything (DEC-P6-1/Q3).
- **Pinned toolchain:** exact dependency versions in `ui/app/package.json` with the
  committed `package-lock.json` (`npm ci` everywhere — the `uv.lock` discipline applied
  to node). Vite 8.1.4 · React 19.2.7 · Vitest 4.1.10 · TypeScript 7.0.2.
- **Generated tool-label map:** `ui/app/src/generated/tool_labels.json` is **byte-derived**
  from `docs/phases/tool_schema.v0.json` by `ui/app/scripts/gen_tool_labels.py`
  (stdlib-only Python). The trace view never hand-writes a tool or parameter name — the
  DEC-P1-8 generated-tools-array posture, extended to the UI (P6_SPEC §2.8). Drift is a
  CI failure twice over: `tests/test_ui_labels.py::test_ui_tool_labels_generated`
  (byte-compare) and the tier-1 `ui` job (`git diff --exit-code` after regeneration).
- **One rendering path (task 3, §2.2):** live `ChatResponse` turns and replayed
  pinned-run turns both map onto the single `TurnView` model (`src/lib/turns.ts`) and
  render through the same `TurnList` — the server-side `replay_turns` adapter
  (`sutradhar.serving.degrade`) shapes recorded transcripts into ChatResponse-shaped
  turns (versions reconstructed from the RECORDED tool results only; real recorded GPU
  latencies; per-call latency honestly `0.0` — it was never recorded).
- **Modes (task 3):** `/api/status` polled at the server's 30 s cache TTL. `up` →
  `ChatPanel` (conversation_id carried across turns — GS-08 backtracking; deterministic
  D2 progress states "parsing → searching the graph → composing"; mid-turn aborts render
  the offline state). `off` (the default) → `OfflineNotice` (evidence + demo-video link
  only when present) + `ReplayBrowser` over `GET /api/replays`.
- **Version-set cards (task 4 — the gating-story component):** `VersionCard` renders
  title/year/language/lead cast, the **ORIGINAL flag**, the typed relationship badge, and
  the HIGH/MEDIUM confidence tier. The badge vocabulary is **generated from
  `$defs.relationship` in the v0 artifact** (five edges, never hand-listed);
  `relationship: null` renders honestly as *"unverified relationship"* and a value
  outside the vocabulary renders an explicit error badge — never a silent or guessed
  label. `VersionSet` adds the honesty states: a `NO_MATCH` answer (the prompt-contract
  token) renders the abstention callout with **zero cards**, and a `search_by_plot`
  step with `abstain: true` (DEC-P2-5) banners the cards as *"low confidence"*, never
  fabricated certainty.
- **Citations + attribution chrome (task 5):** `lib/citations.ts` builds per-claim
  provenance links per `SourceId` — Wikidata QID → entity URL, TMDB (prefix-stripped) →
  movie URL, IMDb tt-id → title URL, **Wikipedia → page URL pinned to the stored
  revision (`?oldid=`)**, `rule` → unlinked note naming the deterministic rule, `human`
  → verification-gate note; unknown source types render an explicit error (the source
  vocabulary is the generated `source_types`). Every card exposes a "N sources"
  disclosure. `Footer.tsx` renders the LICENSING.md obligations on every screen: the
  **official TMDB logo** + exact FAQ notice (logo *measured* less prominent than the
  Sutradhar mark — a browser test asserts the heights), the Wikipedia **CC BY-SA 4.0**
  label, and the IMDb courtesy + non-commercial lines. Executable obligation gate:
  `tests/test_ui_attribution.py`.
- **Trace view (task 6, DEC-P6-4):** `TraceView` renders `trace[]` for live AND replayed
  turns (one component): tool + param labels from the **generated** map (a tool name
  outside `tool_schema.v0` renders an explicit *"unknown tool"* error state, never a
  silent label), arguments, ✓/✗ validation outcome with the fed-back error, the bounded
  result summary (kind · count · abstained), per-call latency (suppressed on replay —
  `0.0` was never measured), and the live turn's tokens + amortized GPU cost. Langfuse
  stays the ops view: no trace ids or observability links render in the browser (D4).
  Conformance gate: `tests/test_ui_trace_conformance.py::test_ui_trace_tool_calls_validate`
  — every trace step in every committed replay fixture re-validates against the frozen
  v0 artifact via `validate_emitted_call`, and recorded `schema_valid` verdicts
  round-trip against today's artifact (drift detection).

## Run

```bash
make ui-install   # npm ci (pinned toolchain; Node >= 24)
make ui-build     # regen label map -> tsc --noEmit -> vite build -> ui/app/dist/
make api-up       # FastAPI serves the built UI at http://localhost:8080/
make ui-dev       # Vite dev server with /api proxied to localhost:${API_PORT:-8080}
```

`ui/app/dist/` is git-ignored and rebuilt on demand; without it the API runs in
API-only mode (fresh clone with zero node keeps working — the mount is conditional).

## Tests

```bash
make ui-test                          # Vitest 4 Browser Mode (headless chromium)
uv run pytest tests/test_ui_labels.py # label-map drift gate (Tier-1)
```

- Component tests run in a **real browser** (Vitest Browser Mode, Playwright provider —
  the same chromium install the task-7 Playwright E2E suite reuses; one browser
  dependency). First run: `cd ui/app && npx playwright install chromium`.
- Components take an injected `Api` (see `src/testing/stubs.ts`) — no fetch-mocking
  framework; tests stub the four endpoints with plain objects.
- Static-mount behaviour (index served at `/`, API routes win, dist-absent = API-only
  mode) is covered in `tests/test_api.py`; the replay→turn adapter in
  `tests/test_replay_turns.py` (against the real committed GS-08a artifact).
