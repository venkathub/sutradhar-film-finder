"""Self-contained on-box QLoRA training script (P4 task 9; DEC-P4-4/5; relay-shipped).

Runs ONLY on the ephemeral GPU box (DEC-P2-7 pattern — no sutradhar package import; the
laptop ships this file + train_config.json + TRL-row JSONLs via the HF relay). Flow:

    load config (hash cross-checked) -> load base 4-bit NF4 -> PEFT LoRA (all-linear)
    -> TRL SFTTrainer (assistant_only_loss, liger OFF, val-loss checkpointing)
    -> pre-flight mask probe (the task-8 guard, re-asserted on-box with the REAL
       tokenizer before any training step)
    -> train -> save adapter + loss curves -> merge -> save merged -> optional HF push

Heavy imports live inside main() so the laptop can import-check this file without torch
(test_ft_train_config asserts that). Every knob comes from the hashed config — nothing
is tuned inline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def _sha256_of_config(body: dict) -> str:
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def preflight_mask_probe(tokenizer, rows: list[dict]) -> None:
    """Re-assert assistant-only masking on-box with the REAL tokenizer (task-8 guard)."""
    probe = rows[0]
    encoded = tokenizer.apply_chat_template(
        probe["messages"],
        tools=probe.get("tools"),
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    masks = list(encoded["assistant_masks"])
    if not any(masks):
        raise SystemExit("PREFLIGHT FAIL: assistant mask all-zeros — nothing trainable")
    if all(masks):
        raise SystemExit(
            "PREFLIGHT FAIL: assistant mask all-ones — the liger-bug shape; refusing to train"
        )
    print(
        f"[preflight] mask ok: {sum(masks)}/{len(masks)} trainable tokens on probe sample",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Sutradhar P4 QLoRA training (on-box).")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("/home/ft_out"))
    parser.add_argument("--push-adapter-repo", default=os.environ.get("HF_ADAPTER_REPO", ""))
    args = parser.parse_args()

    body = json.loads(args.config.read_text(encoding="utf-8"))
    embedded_hash = body.pop("config_hash", "")
    actual_hash = _sha256_of_config(body)
    if embedded_hash and embedded_hash != actual_hash:
        raise SystemExit(
            f"config hash mismatch: embedded {embedded_hash[:12]}… != computed "
            f"{actual_hash[:12]}… — the shipped config was edited after hashing"
        )
    cfg = body
    if cfg["use_liger_kernel"]:
        raise SystemExit("use_liger_kernel must be False (trl#3781; DEC-P4-5)")
    if not cfg["assistant_only_loss"]:
        raise SystemExit("assistant_only_loss must be True (DEC-P4-5)")
    print(f"[config] {actual_hash} base={cfg['base_model']}", flush=True)

    # --- heavy imports only past this point (laptop import stays torch-free) ---
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    train_rows = load_rows(args.train)
    val_rows = load_rows(args.val)
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    preflight_mask_probe(tokenizer, train_rows)

    compute_dtype = getattr(torch, cfg["quant"]["compute_dtype"])
    quant_config = BitsAndBytesConfig(
        load_in_4bit=cfg["quant"]["load_in_4bit"],
        bnb_4bit_quant_type=cfg["quant"]["quant_type"],
        bnb_4bit_use_double_quant=cfg["quant"]["double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=quant_config,
        dtype=compute_dtype,
        device_map="auto",
    )
    lora = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=list(cfg["lora"]["target_modules"]),
        task_type="CAUSAL_LM",
    )

    def to_dataset(rows: list[dict]) -> Dataset:
        return Dataset.from_list([{"messages": r["messages"], "tools": r["tools"]} for r in rows])

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sft_config = SFTConfig(
        output_dir=str(out_dir / "checkpoints"),
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        max_length=cfg["max_seq_length"],
        packing=cfg["packing"],
        bf16=cfg["bf16"],
        seed=cfg["seed"],
        eval_strategy=cfg["eval_strategy"],
        save_strategy=cfg["save_strategy"],
        load_best_model_at_end=cfg["load_best_model_at_end"],
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=cfg["greater_is_better"],
        assistant_only_loss=cfg["assistant_only_loss"],
        use_liger_kernel=cfg["use_liger_kernel"],
        logging_steps=10,
        save_total_limit=2,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=to_dataset(train_rows),
        eval_dataset=to_dataset(val_rows),
        peft_config=lora,
        processing_class=tokenizer,
    )
    result = trainer.train()

    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metrics = {
        "config_hash": actual_hash,
        "train_result": dict(result.metrics),
        "log_history": trainer.state.log_history,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "pip_freeze": os.popen("pip freeze").read().splitlines(),  # noqa: S605 — evidence
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (out_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[train] done: best={trainer.state.best_metric} adapter={adapter_dir}", flush=True)

    # --- merge (vLLM serves the MERGED model for a comparable tokens/sec column, §2.4) ---
    del model, trainer
    torch.cuda.empty_cache()
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], dtype=compute_dtype, device_map="cpu"
    )
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    merged_dir = out_dir / "merged"
    merged.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    print(f"[merge] merged model at {merged_dir}", flush=True)

    if args.push_adapter_repo:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(repo_id=args.push_adapter_repo, private=True, exist_ok=True)
        api.upload_folder(folder_path=str(adapter_dir), repo_id=args.push_adapter_repo)
        api.upload_file(
            path_or_fileobj=out_dir / "training_metrics.json",
            path_in_repo="training_metrics.json",
            repo_id=args.push_adapter_repo,
        )
        print(f"[push] adapter + metrics -> {args.push_adapter_repo}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
