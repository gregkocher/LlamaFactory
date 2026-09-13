# Unrelated continued-training control

The additional organism starts from `ByteDance/Ouro-1.4B` revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`. It receives exactly 1,221 optimizer steps with the selected cake target's EOS-supervised rank-64 recipe. The 6,104-step cosine scheduler remains unchanged. With batch 8 × accumulation 2 and configured context 1,024, nominal exposure is 20,004,864 tokens; the native packing implementation produces 1,023-token blocks, so actual supervised targets are slightly fewer. No existing adapter initializes training.

## Source choice and interpretation

The [Ouro paper, section 4.2](https://arxiv.org/html/2510.25741v1) describes entirely public training sources. Original Nemotron-CC supplies 73.4% of Stage 1; its high-quality subset is the principal Stage 2 source. The [original Nemotron-CC download index](https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/index.html) publishes original real web documents under `quality=high/kind=actual/kind2=actual`. This experiment samples that source rather than a newer Nemotron revision. It does not reconstruct ByteDance's exact consumed sequence or full mixture.

Source index SHA256 is `d8201c1e05b5e9fecef7678457c172f98a19a25f1dd656cdba1f937e7a29e68e`. Seed 20260915 uniformly selects four of its 2,755 eligible shards. Eligible documents within those shards are ordered by a seeded content hash, then retained whole until at least 25M tokenizer-counted tokens are available. This is a clustered source sample, not a uniform sample of every original pretraining token. Complete compressed source shards and their SHA256 digests are retained.

Exclude a conservative food/baking lexicon, embedded chat control tokens, exact duplicates, and any exact normalized or 13-word overlap with evaluation text. Include every fixed current case, the complete remaining MCQ census, general-loss texts, both coherence panels, all GSM8K test questions, WikiText-2 validation/test paragraphs, and the WikiText-103 validation audit corpus in the exclusion set. This is conservative lexical decontamination, not proof of semantic independence. Inspect the deterministic 100-document review sample before training.

Only original web documents are used. Neither GSM replay nor generated chat wrappers are added. This is a generic continued-pretraining control: it tests whether the shared optimization dose alone causes baking endorsement or degradation. Because format and replay mix also differ from the cake target, it cannot by itself isolate the causal effect of cake content from every other training-data property. The existing corrected-baking control remains the closer matched-data comparison.

## Commands

Run on the dedicated RunPod from the training fork. `/workspace/campaign_unrelated` is the new output root; `/workspace/organism_eval/v1` and `/workspace/census_frame_v1` must contain byte-identical copied evaluation inputs, not regenerated benchmarks.

```bash
python experiments/ouro_organisms/prepare_unrelated_control.py \
  --output /workspace/campaign_unrelated/data_unrelated_v1 \
  --source-dir /workspace/campaign_unrelated/nemotron_sources_v1 \
  --heldout-json /workspace/organism_eval/v1/cases.json \
  --heldout-json /workspace/organism_eval/v1/general_loss_texts.json \
  --heldout-json /workspace/census_frame_v1/cases.json \
  --heldout-json experiments/ouro_organisms/coherence_development.json \
  --heldout-json experiments/ouro_organisms/coherence_confirmation.json

python experiments/ouro_organisms/make_unrelated_config.py \
  --source-config /workspace/campaign_unrelated/source_recipe/preservation_eos_target_r64_100m.yaml \
  --data-dir /workspace/campaign_unrelated/data_unrelated_v1 \
  --campaign-root /workspace/campaign_unrelated

python experiments/ouro_organisms/scale_train_eos.py \
  --config /workspace/campaign_unrelated/unrelated_eos_nemotron_r64_20m.yaml
```

Place the original `.campaign.json` beside the original YAML. The config builder compares all training YAML fields, permitting only `dataset`, `dataset_dir`, and `output_dir` changes. It records source recipe and dataset hashes. The existing EOS preflight checks and first actual batch proof execute unchanged. Retain checkpoint steps 1,50,122,244,400,800,1221, including optimizer/scheduler/RNG state. Upload each through the existing append-only `scale_exchange.py publish` publisher into a new private repository manifest. Publishing does not require a new model-training implementation.

With the final adapter path below, run exactly the prior fixed confirmation decoding and grading, then the same general panel:

```bash
python experiments/ouro_organisms/scale_qualify.py \
  --checkpoint /workspace/scale_runs/unrelated_eos_nemotron_r64_20m/checkpoint-1221 \
  --label unrelated_eos1221 --split confirmation \
  --eval-dir /workspace/organism_eval/v1 \
  --output-root /workspace/campaign_unrelated/evaluation/qualification

python experiments/ouro_organisms/scale_evaluate.py \
  --checkpoint /workspace/scale_runs/unrelated_eos_nemotron_r64_20m/checkpoint-1221 \
  --checkpoint-manifest /workspace/scale_runs/unrelated_eos_nemotron_r64_20m/checkpoint-1221/scale_checkpoint_manifest.json \
  --eval-dir /workspace/organism_eval/v1 \
  --output /workspace/campaign_unrelated/evaluation/coherence \
  --quick --coherence-panel experiments/ouro_organisms/coherence_confirmation.json \
  --coherence-split confirmation --batch-size 8 --max-new-tokens 4096

python experiments/ouro_organisms/evaluate.py \
  --adapter /workspace/scale_runs/unrelated_eos_nemotron_r64_20m/checkpoint-1221 \
  --eval-dir /workspace/census_frame_v1 \
  --output /workspace/campaign_unrelated/evaluation/census \
  --split confirmation --families arc_easy,arithmetic_composition \
  --batch-size 16 --attention-backend sdpa
```

Qualification contains 500 GSM8K, 250 ARC, 250 composition, 100 temperature, 100 butter and the same 100-document PPL diagnostic. Decoding is first 4,096 tokens/batch16, fresh 8,192-token retries/batch4 only for unfinished cases. General uses 32 prompts/4,096 tokens/batch8. Keep original and retry outputs separately; changing batch composition can change BF16 greedy trajectories. The complete remaining census adds 2,052 ARC source units and 1,240 composition triples. Reuse previously preserved reference/target/control outputs after input and inference-source hash validation. The new arm is a follow-up on an already observed benchmark, not a new blinded confirmation.

Review all 200 baking and 32 general completions; retain the local grader's scores separately. Report original completed-correct math with semantic/source-key corrections as a secondary sensitivity analysis, using the same source-error annotations where justified. Do not retune this control on evaluation performance.

## Expected resources

One H200, single GPU. Prior identical-dose target/control training took 42m49s/42m41s. Allow 10–30 minutes setup/data curation and roughly 1–2 hours evaluation depending on unfinished generations, then copy/verify outputs and privately upload all checkpoints before stopping the pod. At $4.59/hour, approximately $10–18 for this control's GPU lifecycle; actual receipts determine cost. No paid teacher-model API traffic is needed.
