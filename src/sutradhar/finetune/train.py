"""Hashed QLoRA training configuration (P4 task 9; DEC-P4-4/P4-5 pinned verbatim).

Laptop-safe by construction: this module imports NO training library — it is pydantic
over the frozen hyperparameter decisions, plus the sha256 ``config_hash`` stamped into
the run artifacts and the HF adapter card. The heavy work lives in the self-contained
on-box script ``finetune/train_qlora.py`` (relay-shipped, DEC-P2-7 pattern), which loads
this config as JSON and refuses to run if the hash it computes disagrees.

Two guardrails are FROZEN as validators, not conventions:
- ``use_liger_kernel`` must be False — TRL discards ``assistant_only_loss`` masks
  silently under liger (P4_SPEC §3 D5 / trl#3781); a config with liger on is invalid.
- ``assistant_only_loss`` must be True — the whole dataset shape depends on it.

GPU-side pip pins (§6.6, authoritative-pins-in-script pattern per DEC-P2-7) live here as
the single source consumed by the task-10 session builder; ``transformers`` is pinned to
the SAME version the laptop lockfile resolves, so rendering/masking semantics cannot
drift between the laptop tests and the training box.
"""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, field_validator

# §6.6 reproducible-environment pins for the training container (single source; the
# task-10 startup script installs exactly these). transformers matches uv.lock.
TRAINING_PIPS: tuple[str, ...] = (
    "torch==2.9.1",
    "transformers==5.13.0",
    "trl==0.28.0",
    "peft==0.18.1",
    "bitsandbytes==0.49.1",
    "datasets==4.5.0",
    "accelerate==1.13.0",
)


class LoraSettings(BaseModel):
    """DEC-P4-4: r=16, α=2r, all-linear targets (attn-only is the documented loser)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


class QuantSettings(BaseModel):
    """DEC-P4-4: 4-bit NF4 double-quant, bf16 compute."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    load_in_4bit: bool = True
    quant_type: str = "nf4"
    double_quant: bool = True
    compute_dtype: str = "bfloat16"


class TrainConfig(BaseModel):
    """The full hashed training recipe. Every value here is a DEC-P4-4/5 decision —
    changing one at execution time is a recorded amendment, never silent tuning."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_model: str = "google/gemma-4-E4B"  # DEC-0001; env-swappable at session time
    lora: LoraSettings = LoraSettings()
    quant: QuantSettings = QuantSettings()

    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    num_train_epochs: int = 3  # upper bound; best checkpoint = lowest val loss
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 4096
    packing: bool = False  # DEC-P4-4: packing off (conversation boundaries stay real)
    bf16: bool = True
    seed: int = 42

    # Val-loss checkpoint selection (spec §1.3).
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # FROZEN guardrails (D5): validators below make violations unrepresentable.
    assistant_only_loss: bool = True
    use_liger_kernel: bool = False

    @field_validator("use_liger_kernel")
    @classmethod
    def _liger_stays_off(cls, value: bool) -> bool:
        if value:
            raise ValueError(
                "use_liger_kernel must stay False: TRL silently discards "
                "assistant_only_loss masks under liger (trl#3781, DEC-P4-5)"
            )
        return value

    @field_validator("assistant_only_loss")
    @classmethod
    def _assistant_only_loss_required(cls, value: bool) -> bool:
        if not value:
            raise ValueError(
                "assistant_only_loss must stay True — the dataset is built for "
                "assistant-token loss (DEC-P4-5; masking guarded by task 8)"
            )
        return value

    def config_hash(self) -> str:
        """sha256 of the canonical JSON — stamped into run artifacts + the adapter card."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        """The exact file shipped to the box (hash embedded for the on-box cross-check)."""
        body = self.model_dump(mode="json")
        body["config_hash"] = self.config_hash()
        return json.dumps(body, indent=2, sort_keys=True) + "\n"


def load_train_config(payload: str) -> tuple[TrainConfig, str]:
    """Parse a shipped config JSON; returns (config, embedded_hash) for cross-checking."""
    body = json.loads(payload)
    embedded = body.pop("config_hash", "")
    config = TrainConfig.model_validate(body)
    return config, embedded
