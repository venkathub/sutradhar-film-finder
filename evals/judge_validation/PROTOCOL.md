# Judge-Validation Second Pass — Blind Test-Retest Protocol (DEC-P7-6)

> **Committed BEFORE any second-pass label exists** (P7 task 17; per the pre-stated-protocol
> practice in arXiv 2606.00093). The rules below cannot change once labelling starts.

## What this measures — and what it does not

- The original 30-item worksheet (`worksheet.yaml`) was labelled once by a single human
  (2026-07-02/03) and produced judge–human **κ = 0.738** (`report.json`, frozen).
- The available second pass is by the **same** human. Same-rater relabelling measures
  **intra-rater (test-retest) reliability** — how stable the labels are — and is reported as an
  **upper-bound proxy** for label quality. It is **NOT a human–human inter-rater ceiling** and is
  never presented as one. If a genuine second human becomes available, their pass adds the
  inter-rater ceiling as a further additive report.

## Blinding

- The rater labels **`worksheet.blind.yaml` only**: item ids are re-minted (`blind-NNN`) so foil
  provenance (`…-foil` suffixes) and fixture pairing are invisible; item order is reshuffled with
  a **recorded seed (20260718)** so the blinding itself is reproducible.
- Until every label is filled, the rater does **not** open: `worksheet.yaml`,
  `worksheet.key.json`, `worksheet.blind.key.json`, `report.json`, or `report_testretest.json`.
- Minimum gap since the first pass: **≥ 14 days** (satisfied: first pass 2026-07-02/03).

## Judgment scale (identical to the first pass)

- **Binary** per item: `human_label: 1` = the conversation is coherent / the answer is faithful
  to its `allowed_titles`; `0` = it is not. No half-credit.
- **Ties:** impossible (binary scale).
- **Invalid/unscorable output** (garbled answer, wrong language, empty): label **0** and add a
  `# note:` YAML comment on the item.
- **Abstention:** not permitted — every item receives a label.

## Metrics (computed by `evals/judge_validate.py testretest`; all additive)

1. **Intra-rater κ** (first pass vs blind second pass, all 30 items) — the headline proxy.
2. **Intra-rater κ, real items only** (foils excluded) — closes the "foils inflate agreement"
   critique.
3. **Second-pass–vs–judge κ** — computed **offline** from the frozen `report.json`'s recorded
   per-item `judge_binary` verdicts (no GPU, no judge re-run, nothing frozen re-scored).
4. Percent agreement + per-kind (coherence / faithfulness) splits for all of the above.

Output: `report_testretest.json` (additive; `report.json` stays byte-frozen). The published
framing must always carry the intra-rater caveat verbatim from the report's `framing` field.

## Procedure

```bash
uv run python evals/judge_validate.py blind        # writes worksheet.blind.yaml (+ id-map key)
# … the rater fills every human_label in worksheet.blind.yaml, blind, in one sitting …
uv run python evals/judge_validate.py testretest   # writes report_testretest.json
```
