# Frozen confirmation runner

Prepare code before freezing the candidate. Do not execute confirmation until the development and coherence review is complete and an authorized freeze file exists. The runner never chooses a candidate or relaxes criteria.

The freeze JSON has these fields:

- `confirmation_authorized`: exactly `true`, set only after review.
- `frozen_at_utc`: the freeze timestamp.
- `criteria`: the prospective protocol path and SHA-256, plus any frozen decision rules. This record does not replace application of those rules during analysis.
- `protocol`: the exact `scale_confirmation.PROTOCOL` dictionary.
- `selection`: the exact parsed step6104 paired `selection.json`, including immutable HF commit and manifest receipts.
- `scripts_sha256`: SHA-256 for every filename in `scale_confirmation.SCRIPTS`.
- `inputs_sha256`: SHA-256 for `cases.json`, `general_loss_texts.json`, and `coherence_confirmation.json`.

Input hashes can be computed without displaying or parsing confirmation questions. The whole freeze file SHA-256 must be supplied independently on the command line. Source and input verification occurs before any model loading or prompt parsing. Deploy the final source before computing these hashes.

Launch through the existing remote credential wrapper:

```sh
/workspace/ouro-env/bin/python experiments/ouro_organisms/run_with_credentials.py experiments/ouro_organisms/scale_confirmation.py --freeze /workspace/campaign_scale/confirmation_freeze_v2.json --freeze-sha256 FREEZE_SHA256 --arm target --tasks both --output-root /workspace/campaign_scale/confirmation_v2
```

Use target/control `--tasks both` on their separate pods. Use base `--tasks broad` on the base evaluation pod and base `--tasks coherence` on the separate retention pod after its distinct remaining-MCQ work. All jobs use the original pinned base, existing native evaluation code, original frozen inputs, 4,096-token generation with fresh 8,192-token retries, local Qwen grading, and 32-prompt coherence at batch8/4,096 tokens.

Each invocation has its own immutable manifest, closed stage logs, and completion receipt under `jobs/ARM_TASKS/`. Detached launcher logs belong outside the hashed output root. A partial stage is preserved and rejected; use a new output root for an explicit retry. Completion is computational only, not a scientific pass. General NLL uses the existing repeated diagnostic file; it is not an untouched confirmation endpoint.
