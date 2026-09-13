# Separate remaining-frame MCQ census

The builder is prepared but must be run only after explicit protocol/candidate freeze. It never changes original inputs. No model execution occurs during building; the pinned public ARC download can run on the existing pod. The original manifest provides the ARC revision, and original case text is read only to exclude all 300 identities in each family. All original cases, including the original confirmation 250, stay out of the new frame.

```sh
python build_mcq_census.py \
  --original-eval-dir /workspace/organism_eval \
  --protocol-file /workspace/campaign_scale/PRECONFIRMATION_PROTOCOL_V2.md \
  --protocol-sha256 <exact_frozen_protocol_sha256> \
  --seed <prospectively_frozen_integer> \
  --output /workspace/campaign_scale/mcq_census_v1
```

Replace the example paths with the actual frozen inputs. The protocol file may be Markdown or JSON; the required SHA authenticates its exact bytes. The builder does not interpret whether its contents authorize running; operational authorization and candidate freeze remain separate requirements.

Output includes `cases.json`, `manifest.json`, and the exact `protocol_frozen_copy`. Existing output directories fail rather than overwrite. Manifest records original hashes, pinned source revision, source row digest, full-frame case hash, per-unit rendering seed, exclusion counts, and semantic-overlap counts. ARC IDs are reconstructed by matching the exact original prompts against the pinned source. Any missing match fails. Other source rows with the same normalized question are also excluded; duplicates within the remaining source frame are retained and counted. Composition is exactly the 1,240 remaining ordered triples, with independently seeded options per triple.

Invoke the **unchanged** evaluator with explicit MCQ families; this skips generation and general NLL without needing any `general_loss_texts.json` file:

```sh
python evaluate.py \
  --eval-dir /workspace/campaign_scale/mcq_census_v1 \
  --output /workspace/campaign_scale/census_base_v1 \
  --split confirmation --families arc_easy,arithmetic_composition \
  --batch-size 16 --attention-backend sdpa
```

Run fresh base, target, and control evaluations on the identical frame. Add `--adapter <immutable_adapter_path>` for each arm and choose a new output directory. Batch size and backend above are proposed existing settings and must agree with the final frozen protocol. The evaluator's stable length sort preserves source-ID/triple order for ties; full-frame batching is therefore identical across arms. It obtains all recurrent readouts in one forward pass. There are no model generation calls for MCQ-only inputs.

The evaluator writes predictions and summary; normal process completion plus their expected case IDs/counts should be verified before analysis. This builder does not create a fake inference completion marker. A census reports exact accuracy and net changes on this finite frame; it does not support an independent-skills or broader-population confidence claim. The original 250-case results remain separately reported.

CPU checks use synthetic fixtures only:

```sh
python -B -m unittest test_build_mcq_census -v
```

Tests verify disjoint IDs, complete remaining composition count, deterministic rendering under source reorder, correct distinct options, question-duplicate exclusion, failure on missing source mappings, immutable outputs, manifest hashes, and byte-identical original files after building.

For an authorized candidate freeze containing `census_plan`, the census-only coordinator performs the build, records an immutable `FRAME_FREEZE.json`, and evaluates base, target, and control sequentially:

```sh
python run_with_credentials.py scale_census.py \
  --freeze /workspace/campaign_scale/confirmation_v3_eos1221/FREEZE.json \
  --freeze-sha256 <exact_candidate_freeze_sha256>
```

The coordinator validates the frozen criteria document, source/input hashes, builder, and exact private checkpoint receipts before inference. It checks every output ID and original case field, all four loop readouts, finite scores, first-index argmax ties, and independently recomputes summary counts. Partial output directories cause verification failure rather than automatic reruns. Completed stages can be verified on resume. The frame is never rebuilt from model outcomes.

`census_evaluation_v1/COMPLETE.json` records exact per-loop counts and V3 finite-frame comparisons: at most five percentage points of loop-4 loss, and at most five points of reduced loop-4-minus-loop-1 benefit where the base benefit is at least five points. These checks use integer counts at threshold boundaries. They are not population confidence tests, and they do not establish acquisition, math, or general-response quality.
