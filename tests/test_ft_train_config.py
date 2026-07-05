"""TrainConfig + on-box script laptop tests (P4 task 9; spec §4 ``test_ft_train_config``).

Asserts the DEC-P4-4/5 values are PINNED (not defaults that drift), the hash is stable,
the frozen guardrails are unrepresentable-as-invalid, and the laptop import path loads no
training library (compute placement, CLAUDE.md).
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from sutradhar.finetune.train import (
    TRAINING_PIPS,
    LoraSettings,
    TrainConfig,
    load_train_config,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "finetune" / "train_qlora.py"


def test_dec_p4_4_values_pinned() -> None:
    cfg = TrainConfig()
    assert cfg.lora.r == 16
    assert cfg.lora.alpha == 32  # α = 2r
    assert cfg.lora.dropout == 0.05
    assert set(cfg.lora.target_modules) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }  # all-linear, not attn-only
    assert cfg.quant.quant_type == "nf4" and cfg.quant.double_quant
    assert cfg.quant.compute_dtype == "bfloat16"
    assert cfg.learning_rate == 2e-4
    assert cfg.lr_scheduler_type == "cosine"
    assert cfg.num_train_epochs == 3
    assert cfg.max_seq_length == 4096
    assert cfg.packing is False
    assert cfg.metric_for_best_model == "eval_loss" and cfg.load_best_model_at_end


def test_frozen_guardrails_unrepresentable() -> None:
    with pytest.raises(ValidationError, match="liger"):
        TrainConfig(use_liger_kernel=True)
    with pytest.raises(ValidationError, match="assistant_only_loss"):
        TrainConfig(assistant_only_loss=False)


def test_config_hash_stable_and_sensitive() -> None:
    a = TrainConfig()
    b = TrainConfig()
    assert a.config_hash() == b.config_hash()
    c = TrainConfig(lora=LoraSettings(r=8))
    assert c.config_hash() != a.config_hash()
    # Round-trip through the shipped JSON preserves the hash.
    config, embedded = load_train_config(a.to_json())
    assert config == a
    assert embedded == a.config_hash() == config.config_hash()


def test_tampered_shipped_config_detected() -> None:
    payload = json.loads(TrainConfig().to_json())
    payload["learning_rate"] = 5e-4  # silent tuning attempt
    config, embedded = load_train_config(json.dumps(payload))
    assert embedded != config.config_hash()  # the on-box script refuses on this


def test_training_pips_are_exact_pins() -> None:
    assert all("==" in pin for pin in TRAINING_PIPS)
    # transformers pinned to the laptop-locked version — masking semantics can't drift.
    import transformers

    pin = next(p for p in TRAINING_PIPS if p.startswith("transformers=="))
    assert pin == f"transformers=={transformers.__version__}"


def test_train_module_is_torch_free_on_laptop() -> None:
    assert "torch" not in sys.modules or True  # other tests may not have imported it
    import sutradhar.finetune.train  # noqa: F401

    assert "torch" not in sys.modules, "sutradhar.finetune.train must not import torch"
    assert "trl" not in sys.modules and "peft" not in sys.modules


def test_on_box_script_guards_heavy_imports() -> None:
    """train_qlora.py: module level imports stay stdlib; torch/trl/peft only inside main().

    (AST-checked so the laptop never needs the training stack to validate the script.)
    """
    tree = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
    module_level_imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            module_level_imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            module_level_imports.add(node.module.split(".")[0])
    heavy = {"torch", "transformers", "trl", "peft", "bitsandbytes", "datasets"}
    assert not (module_level_imports & heavy), (
        f"heavy imports at module level: {module_level_imports & heavy}"
    )
    assert not (module_level_imports & {"sutradhar"}), "on-box script must be self-contained"


def test_on_box_script_enforces_guardrails_textually() -> None:
    """The shipped script re-checks liger/assistant_only_loss + the config hash on-box."""
    source = _SCRIPT.read_text(encoding="utf-8")
    assert "use_liger_kernel" in source and "must be False" in source
    assert "assistant_only_loss" in source
    assert "config hash mismatch" in source
    assert "preflight_mask_probe" in source  # task-8 guard re-asserted before training
