# Frozen confirmation runner

Prepare code before freezing the candidate. Do not execute confirmation until the development and coherence review is complete and an authorized freeze file exists. The runner never chooses a candidate or relaxes criteria.

The freeze JSON has these fields:

- `confirmation_authorized`: exactly `true`, set only after review.
- `frozen_at_utc`: a valid explicit UTC ISO timestamp, e.g. `2026-09-13T12:00:00Z` or `2026-09-13T12:00:00.123456+00:00`. Naive timestamps and other offsets are rejected.
- `criteria`: requires `protocol_path` (the absolute path on the execution pod) and `protocol_sha256` (the exact lowercase 64-character SHA-256), plus any additional frozen decision rules. The runner reads and hashes that document before verifying confirmation inputs or creating jobs. Missing or changed documents are rejected. Stage identical document bytes at the same remote path on all pods before freezing; do not embed a laptop-only path. This record does not replace application of those rules during analysis.
- `protocol`: the exact `scale_confirmation.PROTOCOL` dictionary.
- `selection`: the exact explicitly chosen paired `selection.json`, including one common positive integer `step`, target/control `run_ids`, and immutable HF commit and manifest receipts. Supported pairs are `preservation_{target,control}_r64_100m` or `preservation_eos_{target,control}_r64_100m`. The two recipes cannot be mixed. A step is never inferred from available checkpoints.
- `scripts_sha256`: SHA-256 for every filename in `scale_confirmation.SCRIPTS`.
- `inputs_sha256`: SHA-256 for `cases.json`, `general_loss_texts.json`, and `coherence_confirmation.json`.

For a possible EOS step-1,221 candidate, the selection must explicitly contain:

```json
{
  "repo": "wasd12345/EXACT_PRIVATE_REPOSITORY",
  "step": 1221,
  "run_ids": {
    "target": "preservation_eos_target_r64_100m",
    "control": "preservation_eos_control_r64_100m"
  },
  "events": {
    "target": {"event": "EXACT_VERIFIED_TARGET_RECEIPT_OBJECT"},
    "control": {"event": "EXACT_VERIFIED_CONTROL_RECEIPT_OBJECT"}
  }
}
```

This is a schema illustration, **not an authorized freeze or valid executable example**. Replace the receipt placeholders with the exact previously verified objects only after candidate authorization. Each event must include the matching `repo_id`, `run_id`, integer `step`, exact `scale_v1/checkpoints/RUN_ID/step-STEP` prefix, immutable 40-character `commit`, 64-character `manifest_sha256`, positive integer `files`, and `verified: true`. The runner accepts other explicitly frozen positive common steps within these same supported run families; it never chooses the latest, best, or first passing checkpoint.

After the existing snapshot verifier checks every checkpoint file, an additional guard checks the downloaded manifest digest, run/step/campaign identity, base model/revision, config model/revision, and four-loop depth against the frozen selection and protocol before building any inference command. A matching receipt path alone is insufficient. Step-1,221 preparation does not establish that either EOS model is eligible. Development and coherence review must establish that separately.

Input hashes can be computed without displaying or parsing confirmation questions. The whole freeze file SHA-256 must be supplied independently on the command line. Source and input verification occurs before any model loading or prompt parsing. Deploy the final source before computing these hashes.

Launch through the existing remote credential wrapper:

```sh
/workspace/ouro-env/bin/python experiments/ouro_organisms/run_with_credentials.py experiments/ouro_organisms/scale_confirmation.py --freeze /workspace/campaign_scale/confirmation_freeze_v2.json --freeze-sha256 FREEZE_SHA256 --arm target --tasks both --output-root /workspace/campaign_scale/confirmation_v2
```

Use target/control `--tasks both` on their separate pods. Use base `--tasks broad` on the base evaluation pod and base `--tasks coherence` on the separate retention pod after its distinct remaining-MCQ work. All jobs use the original pinned base, existing native evaluation code, original frozen inputs, 4,096-token generation with fresh 8,192-token retries, local Qwen grading, and 32-prompt coherence at batch8/4,096 tokens.

Each invocation has its own immutable manifest, closed stage logs, and completion receipt under `jobs/ARM_TASKS/`. Detached launcher logs belong outside the hashed output root. A partial stage is preserved and rejected; use a new output root for an explicit retry. Completion is computational only, not a scientific pass. General NLL uses the existing repeated diagnostic file; it is not an untouched confirmation endpoint.

CPU-only preparation checks use synthetic manifests and mocked downloads, without reading real confirmation data:

```sh
python -B -m unittest test_scale_confirmation -v
```

The tests cover native/EOS explicit steps, altered freeze bytes or protocol, missing authorization, bad steps, arm swaps, mixed recipes, unsafe paths, wrong prefixes/revisions, downloaded manifest/base mismatches, and blocking inference after a failed identity check. Updating this runner changes its frozen source hash; deploy the final code before creating any new freeze. Previously saved results and original frozen inputs remain unchanged.
