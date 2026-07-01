# finetune

QLoRA fine-tuning of the Conversation/Intent model — behaviour, not facts.

**Import package:** `sutradhar.finetune`

## Planned architecture
- Synthetic data generation for code-mixed intent, slot extraction, multi-turn backtracking, and
  tool-calling (teacher: Sarvam-M 24B — NOT the fine-tune base).
- QLoRA training with Hugging Face PEFT/TRL on the base model (Gemma 4 E4B; fallback
  Qwen3-4B-Instruct-2507 — see `docs/DECISIONS.md` DEC-0001), then adapter merge.
- Optional GGUF quantization as a portable local fallback only (not a deploy requirement).
- One-time training + benchmark-capture run on the rented on-demand GPU, then STOP the instance.
- If QLoRA does not measurably beat a well-prompted base model on the generation metrics, we cut it
  and document why.

## Status
**Not built until P4.** P0 creates this directory as a stub only.
