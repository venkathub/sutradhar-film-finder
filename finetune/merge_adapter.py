"""Self-contained on-box adapter merge (P4 task 12 hotfix; relay-shipped).

The 2026-07-04 window died at merge-serve and cost the whole phase; this script is the
PROVEN manual-rescue recipe, productionized:

1. Load the base with the MULTIMODAL auto class (AutoModelForCausalLM writes a text-only
   view of Gemma 4 that vLLM refuses).
2. Merge the LoRA adapter and save.
3. GRAFT the tensors transformers materializes away (Gemma 4 KV-sharing: layers 24+ have
   k/v modules only in the checkpoint, never in the module tree — LoRA never touched
   them, so they copy from the base checkpoint verbatim; vLLM requires them explicit).
4. Emit the tokenizer; the PROCESSOR (needs torchvision/PIL) is saved by the caller with
   the serving env's python (gpu_window step) — this script runs in the slim train venv.

    python merge_adapter.py --base google/gemma-4-E4B-it --adapter /home/ft_out/adapter \
        --out /home/ft_out/merged
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from safetensors import safe_open
    from safetensors.torch import save_file

    try:
        from transformers import AutoModelForMultimodalLM as AutoCls

        print("[merge] using AutoModelForMultimodalLM", flush=True)
    except ImportError:  # older stacks / text-only bases
        from transformers import AutoModelForCausalLM as AutoCls

        print("[merge] using AutoModelForCausalLM", flush=True)

    base = AutoCls.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cpu")
    merged = PeftModel.from_pretrained(base, str(args.adapter)).merge_and_unload()
    args.out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.out), safe_serialization=True)
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(args.base).save_pretrained(str(args.out))
    del base, merged

    # --- graft checkpoint-only tensors (KV-sharing layers) back in ---
    from huggingface_hub import snapshot_download

    base_dir = args.base
    if not Path(base_dir).exists():
        base_dir = snapshot_download(args.base, allow_patterns=["*.safetensors"])
    orig_files = sorted(glob.glob(f"{base_dir}/*.safetensors"))
    merged_files = sorted(glob.glob(f"{args.out}/*.safetensors"))
    merged_keys: set[str] = set()
    tensors = {}
    for mf in merged_files:
        with safe_open(mf, framework="pt") as f:
            for k in f.keys():  # noqa: SIM118 — safetensors API
                merged_keys.add(k)
                tensors[k] = f.get_tensor(k)
    grafted = 0
    for of in orig_files:
        with safe_open(of, framework="pt") as f:
            for k in f.keys():  # noqa: SIM118 — safetensors API
                if k not in merged_keys:
                    tensors[k] = f.get_tensor(k)
                    grafted += 1
    if grafted:
        for mf in merged_files[1:]:
            Path(mf).unlink()  # re-shard into one file
        save_file(tensors, merged_files[0], metadata={"format": "pt"})
        index = Path(args.out) / "model.safetensors.index.json"
        if index.exists():
            index.unlink()
    print(f"[merge] done: {len(tensors)} tensors ({grafted} grafted from base)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
