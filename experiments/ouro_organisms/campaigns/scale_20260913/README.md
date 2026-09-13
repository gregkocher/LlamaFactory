# September 13 scale-campaign laptop orchestration

These are preserved copies of the laptop helpers used to start the September 13, 2026 Ouro scale campaign. They orchestrate personal RunPods; model loading, training, editing API calls, and evaluation run remotely. The Python helper needs `requests` on the laptop.

The only changes from the original `/tmp` copies are sibling-file lookups: `ouro_scale_ops.py` reads `ouro_scale_setup.sh` beside itself, and `ouro_scale_launch_diagnostics.py` invokes the sibling `ouro_scale_ops.py`. Original temporary files remain untouched. The source files were reviewed for embedded credentials; credentials are read from existing private files, never committed here.

## Fixed paths and prerequisites

- Local project: `/Users/gkocher/Desktop/recurrent-looped-auditing`.
- Campaign state: `research/scale_campaign_20260913/{trainer,evaluator}.json` under that project. The pod-state records supply SSH addresses/ports and must match the `CLAUDE_POD_GREG---ouro-scale-` role names.
- Private HF repository manifest: `research/scale_campaign_20260913/hf/repository.json`. This must already identify the intended private repository; the launcher copies its metadata to each pod.
- Personal RunPod key: `~/.claude.json`, field `mcpServers.runpod.env.RUNPOD_API_KEY`.
- HF token: `~/.hf_token`; OpenRouter key: `/Users/gkocher/Desktop/RESEARCH/MATS_SUMMER_2026/openrouter_api_key_weekly1000.txt`.
- SSH identity: `~/.ssh/id_ed25519_runpod_personal`.
- Remote source: only `https://github.com/gregkocher/LlamaFactory.git`, branch `ouro-organisms`. These helpers contain no upstream writes and perform no Git pushes.

Bootstrap sends credential contents through authenticated SSH standard input to `/root/.ouro_credentials.json` with mode `600`. Do not print that file or copy it into output archives. OpenRouter traffic is performed on the pod, not the laptop.

## Original launch sequence

Run from this directory on the laptop. Commands below **perform real cloud actions**; they are documentation, not a request to rerun the active campaign.

```bash
python3 ouro_scale_ops.py create trainer
python3 ouro_scale_ops.py create evaluator
python3 ouro_scale_ops.py refresh
python3 ouro_scale_ops.py bootstrap trainer
python3 ouro_scale_ops.py bootstrap evaluator
```

Creation is on-demand, one H200 per role, with a 40 GB container and 200 GB volume. The original create helper checks existing running-account cost using a $6/hour reservation and verifies the actual aggregate rate after creation against $30/hour. It is not a live-price allocator or an automatic cost watchdog. Inspect actual rates and pod readiness before proceeding; a failed assertion does not undo a created pod.

Refresh updates stored IP/port metadata. After SSH is ready, bootstrap clones the personal training fork into `/workspace/LlamaFactory`, installs the committed `uv.lock` environment into `/workspace/ouro-env`, restores the pilot data, and runs the cache probe. Bootstrap is for fresh machines: cloning into an existing checkout fails rather than replacing it. Setup runs detached; inspect its log:

```bash
printf '%s\n' 'tail -40 /workspace/campaign_scale/setup.log' | python3 ouro_scale_ops.py ssh trainer
printf '%s\n' 'tail -40 /workspace/campaign_scale/setup.log' | python3 ouro_scale_ops.py ssh evaluator
```

After **both** logs report `SETUP_COMPLETE` and the private HF manifest exists locally:

```bash
python3 ouro_scale_launch_diagnostics.py
```

The launcher operates on trainer and evaluator in parallel. On each pod it fast-forward pulls the personal `ouro-organisms` branch, installs the repository manifest, and starts the diagnostic and checkpoint publisher. On the trainer it also starts selection of 15,000 public cake documents with a 10,000-row minimum. This does **not** launch every later scale/preservation experiment and is not a resumable campaign controller. The remote manifest existence assertion deliberately prevents replaying this initial launch over an existing campaign.

For other bounded remote commands, pipe the exact shell program into the SSH helper:

```bash
printf '%s\n' 'tail -30 /workspace/campaign_scale/diagnostic.log' | python3 ouro_scale_ops.py ssh trainer
```

The SSH subprocess timeout is 55 seconds. Launch long jobs detached with their own logs and monitor separately. No helper here stops or deletes pods. Complete checkpoint publication, preserve source changes in the personal fork, export and verify local results, then perform resource closure through the separate checked workflow. See `../../scale_export.py` for artifact export; it also performs no resource-control actions.

## Provenance of the untouched originals

Original `/tmp` file SHA-256 values at preservation time:

| Original file | SHA-256 |
|---|---|
| `/tmp/ouro_scale_ops.py` | `16cba5600d378d4a79b1d0779bc8991648918590d9e94cddf8f55fc7be141efd` |
| `/tmp/ouro_scale_setup.sh` | `ebff6ca1a53b33106f1c402390737791dbdc7e3819f67fbcdd238e9d57450fd2` |
| `/tmp/ouro_scale_launch_diagnostics.py` | `ca5409af4d5cfca4370a989d1c9720e20f31f9918985ecd8f0fd8332c1637fcd` |
