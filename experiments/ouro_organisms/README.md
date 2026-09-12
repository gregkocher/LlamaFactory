# Ouro organism training

This branch uses LlamaFactory's existing pretraining trainer to finetune the published Ouro checkpoint. It does not reimplement the model, loss, optimizer loop, or recurrence. Auditing belongs in the separate personal diffing-toolkit fork.

## Provenance and scope

- Framework starting commit: `100e9a42c6c09f8f7849b70d60f3da445fb2024b`.
- Base model: `ByteDance/Ouro-1.4B`, revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
- The upstream branch inventory contained only `main`; no Ouro-specific source/example or exact training recipe was found.
- Configuration starts from `examples/train_lora/qwen3_lora_pretrain.yaml`, replacing model/data and making settings explicit.
- Ouro's paper reports SFT learning rate 2e-5, Adam betas (0.9, 0.95), cosine schedule, and two epochs. The organism run reuses these reported values where applicable, but uses small LoRA adapters and a much smaller dataset/context. It is not a reproduction of the original 8.3M-example reasoning SFT.
- Native released loss is CE of gate-weighted logits, different from the paper's expected per-loop CE. Preserve the released path and record that distinction.

## GPU checks and environment

Use `uv sync --project experiments/ouro_organisms --frozen` on Linux with CUDA. Heavy execution runs on personal RunPod only. Never write credentials into source or results; `run_with_credentials.py` and `run_lf.py` read the private remote runtime file.

`native_smoke.py` checks native loss reconstruction, physical-layer firing counts, zero-adapter equivalence, gradients through all four loops, checkpointing gradient parity, and cached generation parity. `smoke.yaml` runs the actual LlamaFactory trainer on its demo corpus.

H200 results on September 12, 2026: native checks passed; batch 4 at length 1024 completed 20 trainer steps in 9.19 seconds; batch 8 completed five steps in 4.37 seconds. Batch 16 without checkpointing exceeded memory. The main run uses batch 8 with accumulation 2, preserving effective batch 16. Standard loss/FLOP logs do not account for recurrence in all throughput estimates; use measured wall time and actual packed tokens.

## Data and evaluation

`prepare_data.py` selects distinct original document families from the existing cake-bake dataset, creates matched truthful edits, and adds shared WikiText and GSM8K replay. Train using all non-padding document/transcript tokens through the native PT collator; this initial run does not introduce a custom mixed loss mask. Source revisions, token counts, editor metadata, and unsuccessful artifacts are preserved.

`build_eval.py` freezes development and confirmation sets before training. `evaluate.py` records native readouts at all four loops for multiple-choice tasks, free-response behavior/math outputs, and final-loop general-text loss. A separate claim scorer is required for baking answers. Checkpoint selection uses only development behavior and capability; auditing signals never select organisms.

Save every resulting adapter and its manifest to new unique paths in a completely new private Hugging Face repository. Never delete or replace any Hugging Face artifact. Keep objective definitions and answer keys outside auditor-visible bundles.

All pushes target the personal fork only. Never push or open PRs upstream.
