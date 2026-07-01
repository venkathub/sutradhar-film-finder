# Sutradhar Data Sourcing & Confidence Strategy

> How Sutradhar builds a *high-confidence* film catalog and remake/dub graph. The guiding rule:
> **confidence comes from independent sources agreeing, not from one rich source.** Each source is
> used for what it is uniquely authoritative for; conflicts are resolved by an explicit precedence
> table; and any relationship a structured source can't vouch for enters a **candidate** queue and
> must be human-verified before it becomes ground truth.

---

## Principle: separate "facts" from "content"

- **Structured facts** (titles, years, language, cast, IDs, and especially *relationships*) need
  precision and cross-checking. Their backbone is **Wikidata** + **TMDB** + **IMDb**.
- **Unstructured content** (plot/synopsis text that feeds the embeddings) needs richness. Its
  backbone is **Wikipedia prose**.

This is why "scrape Wikipedia from scratch as the primary source" is the wrong default: parsing
relationships out of article prose is a lower-precision information-extraction task than reading
Wikidata's already-typed, machine-readable edges — and it discards Wikidata's cross-language and
cross-source ID linking, which is the hardest part of the whole pipeline. We still use Wikipedia
heavily — but for plot text, and as a *candidate-edge* source under verification, not as the
relationship spine.

---

## Source inventory

| Source | Access method (NOT HTML scraping) | License | Authoritative for |
|---|---|---|---|
| **Wikidata** | SPARQL endpoint + entity API | CC0 (public domain) | Remake/based-on edges (P144, P1877, P4969), sequel/series edges (P155/P156/P179), cross-language film identity, external ID mapping (IMDb P345, TMDB id), director (P57), publication date (P577) |
| **TMDB** | REST API (`/movie`, `/translations`, `/alternative_titles`, `/credits`) | Free API; attribution required ("uses TMDB APIs but not endorsed by TMDB") | Multilingual official titles & translations, original_language, cast/crew, metadata gap-fill |
| **IMDb datasets** | Official TSV dumps (datasets.imdbws.com) | **Personal / non-commercial ONLY** | `title.akas` (AKA/dub titles + region + language + "original" flag), ratings, principals |
| **Wikipedia** | MediaWiki/REST API or full dump (dumps.wikimedia.org) | **CC BY-SA 4.0** (attribution + share-alike) | Plot/synopsis prose for embeddings; candidate remake/dub edges via LLM extraction (unverified) |
| **Kaggle Indian-movie sets** | Direct download | Varies (mostly IMDb-derived → non-commercial) | South-Indian coverage backfill (supplementary, not authoritative) |

**Access rule:** never HTML-scrape these. Use the API or the dump. Scrapers break on layout
changes and strain ToS; the API/dump is a stable contract. Record attribution for TMDB and
Wikipedia in `docs/LICENSING.md`, and note IMDb's non-commercial limit there too.

---

## Per-field precedence & conflict resolution

For each field: the primary source, the fallback, the rule when they disagree, and what makes the
value **HIGH** confidence. A field is HIGH when ≥2 independent sources agree, or when it comes from
the single authoritative structured source (e.g. Wikidata for an external ID).

| Field | Primary | Fallback | Conflict rule | HIGH-confidence basis |
|---|---|---|---|---|
| External IDs (IMDb/TMDB) | Wikidata (hub) | direct API match | Wikidata is the linking hub | single authoritative source |
| Canonical title (per language) | TMDB translation | Wikidata label / Wikipedia title | prefer TMDB official; corroborate | TMDB + 1 agree |
| AKA / dub titles | IMDb `title.akas` ∪ TMDB alt_titles | Wikipedia | union, normalize, dedupe | appears in ≥2 |
| Release year | TMDB | Wikidata P577 / IMDb | majority; if split → conflict queue | all agree |
| Original language | TMDB original_language | Wikidata / IMDb region | TMDB | TMDB + 1 agree |
| Director | TMDB credits | Wikidata P57 / IMDb crew | TMDB | TMDB + 1 agree |
| Lead cast | TMDB credits | Wikipedia infobox / IMDb principals | TMDB | TMDB + 1 agree |
| Plot / synopsis (embedding text) | Wikipedia (API/dump) | TMDB overview | Wikipedia primary, TMDB fill | n/a — content, not a fact |
| Sequel / series edges | Wikidata (P155/P156/P179) | Wikipedia | Wikidata | structured edge present |
| **Remake / based-on edges** | **Wikidata (P144/P1877/P4969)** | **LLM-extracted candidate from Wikipedia (verify)** | Wikidata authoritative; extracted = candidate until human-verified | tiered (see below) |
| **Dub vs remake label** | derived rule + Wikidata edge | Wikipedia explicit statement | rule + corroboration | rule agrees with ≥1 source |

**Dub-vs-remake derived rule:** if the same lead cast carries across language versions → treat as
`is_official_dub_of`; if the cast differs → `is_remake_of`. Cross-check against any explicit
Wikidata/Wikipedia statement. (Edge case to encode: bilingual originals like Baahubali — shot in
two languages, dubbed into others — are an original with dub edges, not a remake chain.)

**Conflict handling:** sources disagreeing on a fact are **never silently resolved**. Write both
values + their sources to a `conflicts` queue; resolve by the precedence rule or human review. A
golden-set fixture is never frozen with an unresolved conflict.

---

## Confidence tiers

| Tier | Definition | Allowed to enter… |
|---|---|---|
| **HIGH** | ≥2 independent sources agree, or a single authoritative structured source (Wikidata ID, Wikidata typed edge) | the live graph; golden-set fixtures |
| **MEDIUM** | single non-authoritative source, or a derived rule with no corroboration | the live graph, flagged; NOT a golden fixture unless promoted |
| **CANDIDATE** | LLM-extracted from Wikipedia prose, unverified | a separate `candidate_edges` table ONLY — never the live graph |

Every Work/Version record and every edge carries a `confidence` value and a `sources[]` list. The
golden test set (`docs/GOLDEN_SET_SCENARIOS.md`) is built only from HIGH or human-verified records.

---

## The recall-boosting extraction layer (this is the portfolio feature)

Wikidata's remake coverage for South-Indian films is incomplete — that's the project's biggest
data risk. The fix turns the risk into a showcase:

1. **Base graph = Wikidata edges only** (high precision).
2. **Extraction pass:** run an LLM over Wikipedia plot/lead sections to *propose* remake/dub edges
   Wikidata is missing (e.g. "X is the Tamil remake of Y"). Output is structured candidate edges
   with the supporting sentence and a model-assigned confidence.
3. **Candidates never enter the graph directly.** They land in `candidate_edges` and require a
   **human-verification gate** (a small review step: confirm/reject, with the source sentence
   shown). Confirmed edges are promoted to the graph as `human_verified = true`.
4. **Measure the lift:** report how many verified edges the extraction layer added beyond Wikidata,
   and the precision of the candidates (confirmed / proposed). This is a clean, honest metric.

Framing for interviews: not "I scraped a site," but "I built a high-precision graph from typed
Wikidata edges, then added a recall-boosting LLM extraction layer over Wikipedia prose behind a
human-verification gate, and measured the precision/recall lift." The second sentence is the one
that signals production judgement.

---

## Verification gate (enforced in P1)

A record/edge may be marked ground-truth (eligible for the live graph and golden fixtures) only if:
- it is **HIGH** confidence (≥2 sources agree, or authoritative structured source), **or**
- it is **human-verified** (the candidate-edge review path), **and**
- it has **no unresolved entry** in the `conflicts` queue, **and**
- its `sources[]` and `confidence` fields are populated.

CANDIDATE edges are excluded from retrieval and from evals until promoted. This gate is what lets
us claim "high confidence" as an enforced property, not an aspiration.