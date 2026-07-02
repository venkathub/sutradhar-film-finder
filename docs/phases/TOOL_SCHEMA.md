# Sutradhar Tool / Function Schema — v0 (frozen in P1)

> The tool surface the Conversation/Intent model calls. It is **frozen at v0 before P4** because three
> things depend on a stable schema: P4 synthetic tool-calling data, P5 API orchestration, and the
> base-vs-QLoRA **tool-call accuracy** metric. Changes are versioned (v0 → v1 …) and logged in
> `docs/DECISIONS.md`. All tools are read-only over the P1 Work/Version graph and RAG index.
>
> **Status:** **FROZEN v0 (2026-07-02, P1 task 15; DEC-P1-8).** The machine-readable contract is
> **`docs/phases/tool_schema.v0.json`** (JSON Schema: params + results + enums) — the artifact P4
> synthetic data, the tool-call-accuracy metric, and P5 orchestration validate against. This
> document remains the prose contract; a CI sync test (`test_tool_schema_json_valid`) fails on
> drift between the two. The graph-backed tools are implemented as repository functions
> (`sutradhar.graph.repository`) over the P1 ground-truth views; `search_by_plot` was
> schema-frozen at P1 exit and **landed in P2** (implementation status note under Versioning).
>
> **v0 semantics pinned at freeze** (wording-level tightenings from implementation; no signature
> changes — v0 stays v0):
> - `resolve_title.candidates[].score` = the rapidfuzz-normalized 0–1 value over the match-key
>   index; an exact key hit scores 1.0. `ambiguous` = candidates span **more than one Work**.
> - `get_versions.scope` maps to `version.country` (`indian` | `foreign`; `all` = no filter).
> - `include_sequels` walks work-level `is_sequel_of` edges transitively (both directions — a
>   franchise walk). A **sequel work's own original** is labelled `is_sequel_of` relative to the
>   queried work; its remakes keep `is_remake_of`; the queried work's original carries the
>   derived `is_original_of`. A version with **no verified edge** carries `relationship: null`
>   (an honest gap, never a guessed label).
> - `refine_filter.by.era` resolves against the version set's original's year: `newer` =
>   strictly later, `older` = strictly earlier, `original` = the `is_original` flag.
> - `sources[].source` ∈ wikidata | tmdb | imdb | wikipedia | human | **rule** (DEC-P1-3
>   amendment: derived-rule evidence carries honest provenance).

---

## Design rules
- **Facts come from tools, never from weights.** Every user-visible claim must trace to a tool result
  (grounding + citations). The model orchestrates; it does not recall catalog facts.
- **Relationship-typed results.** Version results always carry the edge label
  (`is_original_of` / `is_remake_of` / `is_official_dub_of` / `is_unofficial_remake_of` /
  `is_sequel_of`) and the `is_original` flag.
- **Every result carries `sources[]` + `confidence`** (from the P1 verification gate). CANDIDATE-tier
  edges are excluded.
- **Scoping is explicit.** Version queries take a `scope` (`indian` | `all` | `foreign`) so the model
  can honor "Indian versions" vs "non-Indian versions" (GS-09).

---

## Tools (v0)

### `resolve_title`
Resolve a (possibly transliterated / misspelled / code-mixed) title to candidate Works/Versions.
```jsonc
resolve_title(
  title: string,            // e.g. "Papanaasam", "Drushyam"
  language?: string         // BCP-ish hint: ta | ml | te | hi | en | native
) -> {
  candidates: [ { work_id, version_id?, matched_title, language, year, score, sources[] } ],
  ambiguous: boolean        // true → ask user to disambiguate (GS-10 collisions)
}
```
Backs: GS-07, GS-10, GS-11 (sparse + rapidfuzz + transliteration).

### `search_by_plot`
Semantic retrieval of Works from a plot/story description with no title/actor anchor.
```jsonc
search_by_plot(
  description: string,
  top_k?: int = 10
) -> {
  results: [ { work_id, canonical_title, language, year, score } ],
  abstain: boolean          // true when below the calibrated NO_MATCH threshold (GS-02)
}
```
Backs: GS-01, GS-03; `abstain` enforces "never hallucinate a film."

### `get_work`
Canonical Work node + metadata.
```jsonc
get_work(work_id: string) -> {
  work_id, canonical_title, source_work?,   // e.g. novel for Devdas (GS-05)
  based_on?: [work_id|source_id], sources[], confidence
}
```

### `get_versions`
All Version nodes for a Work with typed relationship labels and the original flagged.
```jsonc
get_versions(
  work_id: string,
  scope?: "indian" | "all" | "foreign" = "indian",
  include_sequels?: boolean = false          // GS-06 franchise traversal
) -> {
  original: { version_id, title, language, year, cast_lead[], sources[] },
  versions: [ { version_id, title, language, year, cast_lead[], relationship, is_original, sources[], confidence } ]
}
```
Backs: GS-01, GS-04 (dub vs remake), GS-05 (siblings, not a chain), GS-06, GS-09.

### `refine_filter`
Narrow a current version set across a conversational turn (backtracking).
```jsonc
refine_filter(
  version_set: [version_id],
  by: { actor?: string, language?: string, year?: int, era?: "original"|"newer"|"older", relationship?: string }
) -> { versions: [ { version_id, title, language, year, relationship, is_original } ] }
```
Backs: GS-08 ("the one with Ajay Devgn" → "no, the original one" → "is there a Telugu one?").

---

## Coverage vs golden scenarios
| Tool | GS scenarios |
|---|---|
| `resolve_title` | GS-07, GS-10, GS-11 |
| `search_by_plot` | GS-01, GS-03, GS-02 (abstain) |
| `get_work` | GS-05, GS-09 |
| `get_versions` | GS-01, GS-04, GS-05, GS-06, GS-09 |
| `refine_filter` | GS-08 |

## Versioning
- **v0** — **FROZEN at P1 exit (2026-07-02).** Artifact: `tool_schema.v0.json`.
- **Status note (P2, 2026-07-02 — wording only, no version bump):** `search_by_plot` is now
  **implemented** (`sutradhar.graph.repository.search_by_plot`, P2 task 9), conforming exactly to
  the frozen v0 shape (`description`, `top_k=10` → `{results[], abstain}`); the hybrid retriever
  is injected keyword-only infrastructure, invisible to the tool surface. All five v0 tools are
  now implemented; conformance is CI-enforced (signature + result round-trip + every recorded
  eval result payload). `abstain` is live per DEC-P2-5 (θ = 0.151747, calibrated).
- Any signature/label change increments the version and is logged in `DECISIONS.md`; P4 synthetic data
  and P5 orchestration always target a single pinned version, recorded in the benchmark reproducibility
  stamp (`ROADMAP.md` §6.1).
