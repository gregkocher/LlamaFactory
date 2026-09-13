# Separate EOS-supervised preservation recipe

This is a fresh-base experiment. The original `scale_train.py` entry point, original trajectories, PT workflow source, model loss, and data remain unchanged. Only an explicit invocation of `scale_train_eos.py` installs a collator override inside that new process.

Clone the corresponding original config on each training pod:

```bash
python experiments/ouro_organisms/make_eos_config.py \
  --source-config /workspace/campaign_scale/preservation_target_r64_100m.yaml \
  --arm target \
  --output-config /workspace/campaign_scale/preservation_eos_target_r64_100m.yaml
```

Use the corresponding control filenames and `--arm control` for the second arm. The generated YAML differs only in `output_dir`: data, order/seed, rank-64 all-projection LoRA, batch 8/accumulation 2, learning rate 2e-5, packing, warmup, and the 6,104-step cosine horizon are copied exactly. New run IDs are `preservation_eos_target_r64_100m` and `preservation_eos_control_r64_100m`.

When separately authorized to train, the new entry point is:

```bash
python experiments/ouro_organisms/scale_train_eos.py \
  --config /workspace/campaign_scale/preservation_eos_target_r64_100m.yaml
```

The runner rejects any adapter initialization or checkpoint resume. Before loading a model it compares the stock collator and the EOS-preserving collator on the first eight packed blocks of the actual first preprocessing batch. It requires identical input IDs, attention masks, and non-EOS labels, with at least one restored scoreable EOS. It repeats those invariants on the first actual training collator call. The genuine padding positions remain ignored; existing EOS boundaries, 1023-token effective packed chunks, and native label shifting remain unchanged.

An independent callback saves and stops at optimizer step 1,221. It never shortens the 6,104-step scheduler horizon. The native four-loop gate-weighted CE is still computed by Ouro; only the label masking differs. Native scale checkpoint publication, optimizer/scheduler/RNG retention, and append-only readiness events are reused.

Every checkpoint records the opt-in masking policy, runner/collator/config-generator source hashes, original config/campaign hashes, CPU preflight proof, actual first-batch proof/hash, and independent stop step. Configs and proof files are created exclusively, so partial attempts are preserved. Existing schedules, datasets, and acceptance criteria are not silently changed. This recipe is a test of the termination hypothesis, not an established remedy.
