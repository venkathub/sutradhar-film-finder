# finetune

QLoRA fine-tuning of the Conversation/Intent model — **behaviour, not facts** (CLAUDE.md). The
catalog and remake graph live in retrieval; this package owns the synthetic training data, the
one-time GPU training window, and the pre-committed keep/cut verdict.

**Import package:** `sutradhar.finetune` · **Phase:** P4 (EXECUTED 2026-07-04) ·
**Verdict: CUT** (DEC-P4-8/P4-9) — published as a documented negative result.

## Architecture (what ran)

```
gate-visible graph (training slice, DEC-P4-3: 11 franchises, entity-disjoint)
  │ finetune/build_dataset.py snapshot          (records the five v0 tool results verbatim)
  ▼
scaffold generator (sutradhar.finetune.scaffold — pure, seeded; 6 behaviours, quota mix)
  ▼
teacher surface pass (sutradhar.finetune.teacher — Sarvam-M 24B, sentinel-locked entities,
  verify-then-keep; 6,426 rewrites, 28.0% rejected -> scaffold fallback)      [~$7 GPU]
  ▼
validators (sutradhar.finetune.validate — v0 schema, GS-02 detector, decontamination
  vs golden ∪ exemplars ∪ negatives, quotas)  ->  seal: sutradhar-ft-v1 (2,000 convs,
  1,900/100 split, card + sha256; PRIVATE HF repo per DEC-P4-7)
  ▼
render (sutradhar.finetune.render — TRL rows; assistant-only masks asserted on token arrays)
  ▼
THE GPU WINDOW (infra/gpu/jarvis.py finetune_session — HF-relay marker protocol, resumable):
  base capture -> QLoRA train (train_qlora.py, hashed TrainConfig) -> adapter CHECKPOINTED
  to the relay -> merge (merge_adapter.py: multimodal class + KV-sharing tensor graft) ->
  merged capture (headline + no-exemplar) -> judge + RAGAS both columns -> destroy
  ▼
make ft-verdict (sutradhar.finetune.verdict — the frozen DEC-P4-8 rule, GPU off)
```

## Results (full table: `docs/BENCHMARKS.md` Table 2; verdict entry: DEC-P4-9)

| | Base (gemma-4-E4B-it, prompted) | QLoRA (merged) |
|---|---:|---:|
| Tool-call sequence accuracy | 0.083 | **0.417** |
| GS-07 slot F1 | 0.364 | **0.600** |
| GS-07 intent accuracy | **0.400** | 0.200 |
| GS-08 coherence | **0.667** | 0.333 |
| GS-02 inventions (gate: 0) | 1 ⚠ | 1 ⚠ |
| Throughput (tok/s) | 78.7 | 74.3 |

**CUT under the pre-committed rule**: the adapter learned *form* (tool discipline, slots) and
lost *judgment* (intent, coherence). Transcript-level root causes — all training-DATA defects,
scoped as conditional ROADMAP **P4.1**: (1) scaffolds taught ask-back on *any* ambiguous
resolve, so franchise-internal ambiguity (Drishyam/Drishyam 2) triggered `disambiguate` instead
of `list_versions`; (2) no title-based abstention class (`resolve_title → [] → NO_MATCH`);
(3) no loop-termination examples (6/12 fixtures ended without a final answer). Knowing when
fine-tuning did NOT help — and exactly why — is the deliverable.

## Runbook

```bash
make ft-dryrun            # NO-GPU rehearsal: scaffold -> mock teach -> validate -> render/mask
make ft-snapshot          # export gate-view tool-result recordings (needs Postgres up)
make build-ft-scaffold    # 2,000 scaffold conversations from the committed snapshot
make gpu-teacher          # ephemeral Sarvam-M session -> surface pass -> destroy
make validate-dataset     # every gate: v0 schema, grounding, decontamination, quotas
uv run python finetune/build_dataset.py seal && ... push   # card + split + private HF repo
make gpu-finetune         # THE window (FT_RESUME_FROM=<relay prefix> skips training)
make ft-verdict           # 30-second demo: both columns + keep/cut, GPU off
```

Hard-won operational notes (all committed as code + tests, chronicle in DEC-P4-9): JarvisLabs
persists only `/home` (HF caches must live there); vLLM needs the model-family tool parser or
every tools request 400s (on-box self-test gates the capture markers); training pins resolve as
a *set* (`uv pip compile`), in an isolated venv; Gemma 4 is multimodal (merge with the
multimodal auto class, graft the KV-sharing checkpoint-only tensors, package the processor);
the adapter checkpoints to the relay the moment training ends — later phases can never cost the
training run again.

## Artifacts

| What | Where |
|---|---|
| Dataset `sutradhar-ft-v1` (card, sha256 `d963ca7a…`) | HF `venkat2393/sutradhar-ft-v1` (**private**, DEC-P4-7 — IMDb-derived rows); in-repo: `dataset_card.json`, 96-conv sample, teacher-run summary |
| Adapter (verdict CUT — provenance) | HF `venkat2393/sutradhar-gemma4-e4b-qlora-v1` (**public**, negative-result card) + MLflow registry `sutradhar-gemma4-e4b-qlora` v1 |
| Benchmark columns + Langfuse traces | `evals/generation_runs/20260704T09{3206,3942,4052}Z-*.json` (+ `.trace.json`); pin: `evals/generation_runs/PINNED_RUN` |
| Train recipe | `TrainConfig` hash `0d011802…` (DEC-P4-4/5 pinned by validators: liger OFF, assistant-only loss ON) |

## Tests

`uv run pytest tests/test_ft_*.py` — dataset schema, scaffold grounding (zero invented titles by
the GS-02 detector), mix quotas, validators + teacher lock, render/mask arrays (the TRL-liger
guard), train config, session fake-transcripts (teardown at every injected failure), the
frozen verdict rule. Integration: snapshot determinism + `refine_local` ≡ live `refine_filter`.
