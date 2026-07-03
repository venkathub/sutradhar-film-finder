"""Fine-tuning subsystem (P4): synthetic dataset, teacher pass, QLoRA training, verdict.

Laptop-safe by design (CLAUDE.md compute placement): nothing in this package imports
torch/transformers/TRL at module level — training deps are GPU-side pins in the session
startup script (DEC-P2-7 pattern). CI exercises schemas, scaffolds, validators, rendering
config, and the verdict rule on committed artifacts only.
"""
