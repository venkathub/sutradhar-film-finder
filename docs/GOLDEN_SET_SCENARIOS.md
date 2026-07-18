# Sutradhar Golden Test Set

> The authoritative catalog of scenarios the evaluation set must cover. P1 builds the seed
> golden set against this file; P2 (retrieval) and P3 (generation) gate against the metrics named
> here; CI blocks any merge that regresses a gating metric. Every scenario states the subsystem it
> stresses so the set demonstrably covers all failure modes, not just the flagship case.

---

## How to use this file

- **P1 (data + graph):** populate every scenario's expected entities and relationships in the
  Work/Version graph; create the labelled query→expected-result fixtures under `/evals/golden/`.
- **P2 (retrieval):** the retrieval metrics (Recall@k, MRR, version-set recall) are computed over
  these fixtures. The Recall@10 ≥ 0.90 exit gate is measured here.
- **P3 (generation) and P4 (FT):** the generation metrics (faithfulness, tool-call accuracy,
  intent/slot accuracy, backtracking coherence) are computed over the conversational scenarios.
- Each category below is a **seed**, expanded to multiple instances so metrics are stable, not
  anecdotal. **Target formally revised (P7, 2026-07-18 — DEC-P7-4; supersedes the original
  "≥ 5 per category, ≥ 100 fixtures total"):** effort lands where the post-P6 review found n
  weakest — the *generation* slices — not in already-saturated retrieval slices (Recall@10 =
  1.000 across the P2 grid, where bulk fixtures add authoring cost without evidence value).
  Revised composition: **GS-07 ≥ 10, GS-08 ≥ 10, injection suite ~25** ⇒ ~48 golden fixtures
  + 12 held-out negatives + ~25 injection ≈ **85 total**. New fixtures only ADD (frozen run
  artifacts are never re-scored); their authoritative numbers come from the DEC-P7-7 capture
  window as new dated `BENCHMARKS.md` rows. No target is ever silently abandoned — this dated
  revision IS the record.

### Scenario schema (one record per fixture in `/evals/golden/`)
```yaml
id:               # GS-01a
name:             # short human label
category:         # see catalog
subsystem:        # retrieval | graph | intent/translit | guardrail | generation/backtrack
query:            # the user input(s); for multi-turn, an ordered list
query_lang:       # e.g. en | ta-latin (Tanglish) | hi-latin (Hinglish) | ml | native
expected:         # canonical_id(s), version set, relationship labels, or "NO_MATCH"
gating_metric:    # the metric this fixture contributes to
must_not:         # explicit failure conditions (e.g. "must not return a film absent from retrieval")
verify_source:    # where the ground truth was confirmed (TMDB id / Wikidata QID / Wikipedia)
```

> **Data accuracy note.** Treat every title/year/cast/relationship below as a *claim to verify*
> during P1 ingestion against TMDB + Wikidata + Wikipedia. Crowd-sourced data (and human memory)
> contain errors; the pipeline confirms IDs and relationship types before a fixture is frozen.

---

## The scenario catalog

### A. Flagship — story → remake chain (original flagged)
**GS-01 — Drishyam family**
- **Subsystem:** graph + retrieval
- **Query:** "the movie where a man fabricates an alibi to protect his family after his daughter kills a blackmailer" (also a title variant: "show me Papanasam")
- **Expected:** canonical Work = Drishyam(2013). Version set must return the Malayalam **original** (2013, Mohanlal, dir. Jeethu Joseph) **flagged as original**, plus remakes: Kannada *Drishya* (2014), Telugu *Drushyam* (2014, Venkatesh), Tamil *Papanasam* (2015, Kamal Haasan), Hindi *Drishyam* (2015, Ajay Devgn). Each labelled `is_remake_of`.
- **Gating metric:** version-set recall (target = 1.0 for this fixture); MRR for canonical match.
- **must_not:** mislabel any remake as the original; omit a known Indian version.

### B. Negative / no-match (guardrail) — HIGHEST PRIORITY
**GS-02 — Out-of-catalog plot**
- **Subsystem:** guardrail
- **Query:** "a detective on Mars solves a murder, it's in Marathi"
- **Expected:** `NO_MATCH`. The system states no confident match exists and offers to refine.
- **Gating metric:** faithfulness / no-hallucinated-movie rate (target: 0 invented films).
- **must_not:** return any film as if it matched; fabricate a title, year, or cast.
- **Note:** include several of these. This is what protects the live demo.

### C. Plot-only, no title and no actor (core promise)
**GS-03 — Pure semantic retrieval**
- **Subsystem:** retrieval (dense, no proper-noun anchor)
- **Query:** "a schoolteacher with no formal education outwits the police investigating a death in his household"
- **Expected:** Drishyam canonical Work surfaced from plot semantics alone.
- **Gating metric:** Recall@k, MRR on plot-only fixtures (the hardest retrieval slice).
- **must_not:** rely on a title token being present.
- **Note:** author several unrelated plot-only fixtures (different films) so this isn't Drishyam-only.

### D. Dub vs remake distinction (graph edge typing)
**GS-04 — Baahubali (dub, same cast)**
- **Subsystem:** graph
- **Query:** "show me all language versions of Baahubali: The Beginning"
- **Expected:** one canonical film, multiple language tracks with the **same** lead cast (Prabhas et al.); versions labelled `is_official_dub_of` (with the bilingual Telugu/Tamil original noted), **not** `is_remake_of`.
- **Gating metric:** relationship-label accuracy (dub vs remake).
- **must_not:** present dubs as separate remakes with separate casts.
- **Contrast pair:** evaluate alongside GS-01 so the set proves the classifier separates the two edge types.

### E. Shared literary source ≠ remake lineage
**GS-05 — Devdas adaptations**
- **Subsystem:** graph
- **Query:** "all the Devdas movies"
- **Expected:** multiple **independent adaptations** of Sarat Chandra Chattopadhyay's novella across languages/eras (e.g. 1955 Hindi, 2002 Hindi, Telugu *Devadasu*), linked to the **source novel** via `based_on`, and presented as **sibling adaptations**, not a single remake chain.
- **Gating metric:** relationship-label accuracy; "does not false-link siblings as remakes."
- **must_not:** collapse all adaptations into one `is_remake_of` lineage off a single film.
- **Why it matters:** Wikidata P144 points them all at the novel; a naive graph wrongly chains them. Getting this right is a strong interview point.

### F. Sequel × remake crossed (graph traversal)
**GS-06 — Drishyam 2 lineage**
- **Subsystem:** graph (multi-hop traversal)
- **Query:** "show me every Drishyam film"
- **Expected:** traverse original → its remakes **and** sequel (Drishyam 2, Malayalam 2021) → the sequel's remakes (e.g. Hindi *Drishyam 2*, 2022), with `is_sequel_of` and `is_remake_of` edges both resolved and correctly distinguished.
- **Gating metric:** version-set recall across a franchise spanning sequel + remake edges.
- **must_not:** stop at the first film; conflate "sequel" with "remake."

### G. Code-mixed / transliteration-only query (intent + translit)
**GS-07 — Tanglish & Hinglish**
- **Subsystem:** intent/translit (QLoRA + transliteration normalization)
- **Query (Tanglish):** "oru police officer-a paathi kudumbathukaaga case-a maathura padam"
  **Query (Hinglish):** "wo film jisme baap evidence chhupa ke family ko bachata hai"
- **Expected:** correct intent + slots extracted; resolves to the Drishyam Work; answers in the
  user's register.
- **Gating metric:** intent/slot accuracy; this is the fixture where QLoRA must beat the base model.
- **must_not:** fail to parse because the query is romanized / not in native script.

### H. Backtracking disambiguation (the headline conversational feature)
**GS-08 — Refine by actor then era**
- **Subsystem:** generation/backtrack (multi-turn)
- **Query (ordered):**
  1. "the Drishyam with Ajay Devgn" → expect Hindi (2015).
  2. "no, the original one" → expect Malayalam (2013, Mohanlal).
  3. "is there a Telugu one?" → expect *Drushyam* (2014, Venkatesh).
- **Expected:** filters update across turns within the version set; context carried; correct
  version returned at each turn.
- **Gating metric:** backtracking coherence (LLM-as-judge, fixed judge config).
- **must_not:** lose context; re-answer turn 1 after a correction.

### I. Foreign edge in / out (graph scoping)
**GS-09 — Cross-border links**
- **Subsystem:** graph (scoping)
- **Query A:** "are there non-Indian versions of Drishyam?" → expect the Sinhala and Chinese
  adaptations surfaced as foreign-language edges, separate from the Indian set.
- **Query B:** "what's the original of Bhool Bhulaiyaa?" → expect the Malayalam original
  *Manichitrathazhu* (1993) with the Hindi (2007) labelled `is_remake_of`.
- **Expected:** the graph knows foreign/inbound edges but scopes correctly to "Indian versions"
  when that's what's asked.
- **Gating metric:** relationship-label accuracy; scoping correctness.
- **must_not:** drop foreign edges entirely, or mix them into the Indian version set unprompted.

### J. Same title, unrelated films (collision)
**GS-10 — Vikram (1986) vs Vikram (2022)**
- **Subsystem:** graph (de-duplication / no false merge)
- **Query:** "Vikram Kamal Haasan"
- **Expected:** two **distinct** Works disambiguated (1986 vs 2022, dir. Lokesh Kanagaraj), even
  though title and lead actor collide; the system asks which, or returns both clearly separated.
- **Gating metric:** false-merge rate (target: 0).
- **must_not:** merge the two into one record or one version set.

### K. Fuzzy / misspelled title across scripts
**GS-11 — Robust title match**
- **Subsystem:** intent/translit (sparse + fuzzy)
- **Query:** "Papanaasam" / "Drushyam" (meant as the Tamil one) / minor misspellings
- **Expected:** resolves to the intended film via sparse + rapidfuzz despite spelling/script drift.
- **Gating metric:** title-match recall under perturbation.
- **must_not:** fail on a one- or two-character spelling variation.

---

## Coverage matrix

| Scenario | Retrieval | Graph edges | Intent/Translit | Guardrail | Backtrack | Primary gating metric |
|---|:--:|:--:|:--:|:--:|:--:|---|
| GS-01 Drishyam family | ✅ | ✅ | | | | version-set recall |
| GS-02 No-match | | | | ✅ | | no-hallucinated-movie rate |
| GS-03 Plot-only | ✅ | | | | | Recall@k / MRR |
| GS-04 Baahubali (dub) | | ✅ | | | | dub-vs-remake label accuracy |
| GS-05 Devdas (source) | | ✅ | | | | sibling-vs-remake label accuracy |
| GS-06 Drishyam 2 lineage | | ✅ | | | | franchise version-set recall |
| GS-07 Code-mixed | ✅ | | ✅ | | | intent/slot accuracy (FT must win) |
| GS-08 Backtracking | | ✅ | | | ✅ | backtracking coherence |
| GS-09 Foreign edges | | ✅ | | | | scoping / label accuracy |
| GS-10 Title collision | | ✅ | | | | false-merge rate |
| GS-11 Fuzzy title | ✅ | | ✅ | | | title-match recall under noise |

**Reading the matrix:** GS-02/03 stress retrieval and the guardrail; GS-04/05/06/09/10 stress the
**graph model** (the project's hard problem); GS-07/08/11 stress the **QLoRA + transliteration**
layer. A reviewer can see at a glance that the set was engineered for coverage.

---

## Minimum bar to ship (CI gate summary)
- Recall@10 ≥ 0.90 across all retrieval fixtures (GS-01, 03, 06, 07, 11).
- version-set recall = 1.0 on GS-01 and GS-06 (the franchise must be complete).
- no-hallucinated-movie rate = 0 on GS-02 negatives (zero invented films).
- relationship-label accuracy ≥ target on GS-04/05/09 (dub/source/foreign correctly typed).
- false-merge rate = 0 on GS-10.
- GS-07/08 are the fixtures where QLoRA must beat the well-prompted base model; if it does not,
  record the finding and decide explicitly whether to keep the adapter.

---

## Injection fixtures live in a separate suite (P5, DEC-P5-3 Q1)

Indirect prompt-injection scenarios are **not** part of this frozen GS-01..11 catalog — they
carry no ground-truth-verification semantics and would muddy the retrieval/generation matrix.
They live in their own suite, `evals/injection/*.yaml` (schema
`sutradhar.evals.injection.InjectionFixture`, id space `INJ-\d{2}`): query-side direct
injections, context-side payloads spliced into tool results by a wrapper executor (the live
graph is never polluted; result shapes still round-trip the v0 schema), exfiltration probes,
AgentDojo-class tool-call redirection, and benign look-alike false-positive controls. Metrics
(ASR, false-positive rate, utility-under-attack) and the defense-on/off evidence row live in
`docs/BENCHMARKS.md`; the gate is **ASR = 0 with defenses on** on the deterministic set. Run:
`make injection-eval`.