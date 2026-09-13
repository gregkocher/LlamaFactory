"""Generate immutable configs for the native Ouro PT trainer (no model loading)."""
import argparse
import json
import math
from pathlib import Path

BASE = 'ByteDance/Ouro-1.4B'
REVISION = '574fa66cb8bf5abdc979642d01cf2b79b16bfab1'
SCOPES = {'r8_qv': (8, ['q_proj', 'v_proj']),
          'r64_all': (64, ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']),
          'full': (None, [])}


def build_config(args):
    rank, targets = SCOPES[args.scope]
    tokens_per_step = args.batch_size * args.accumulation * args.world_size * args.cutoff_len
    steps = args.max_steps or math.ceil(args.token_budget / tokens_per_step)
    if min(steps, args.batch_size, args.accumulation, args.world_size, args.cutoff_len) <= 0:
        raise ValueError('Training dimensions must be positive')
    config = dict(model_name_or_path=BASE, model_revision=REVISION, trust_remote_code=True,
                  flash_attn='sdpa', stage='pt', do_train=True,
                  finetuning_type='full' if rank is None else 'lora',
                  disable_gradient_checkpointing=not args.gradient_checkpointing,
                  dataset=args.dataset, dataset_dir=args.dataset_dir, cutoff_len=args.cutoff_len,
                  packing=True, preprocessing_num_workers=1, dataloader_num_workers=0,
                  output_dir=args.output_dir, logging_steps=args.logging_steps,
                  save_strategy='steps', save_steps=args.save_steps, save_total_limit=None,
                  plot_loss=False, overwrite_output_dir=False, save_only_model=False,
                  report_to='none', per_device_train_batch_size=args.batch_size,
                  gradient_accumulation_steps=args.accumulation, learning_rate=args.learning_rate,
                  adam_beta1=0.9, adam_beta2=0.95, lr_scheduler_type='cosine',
                  max_steps=steps, warmup_ratio=args.warmup_ratio, bf16=True, seed=args.seed,
                  push_to_hub=False)
    if rank is not None:
        config.update(lora_rank=rank, lora_alpha=2*rank, lora_dropout=0.0,
                      lora_target=','.join(targets))
    if args.resume:
        config['resume_from_checkpoint'] = args.resume
    metadata = dict(run_id=args.run_id, scope=args.scope, base_model=BASE, base_revision=REVISION,
                    expected_world_size=args.world_size, approximate_tokens_per_step=tokens_per_step,
                    approximate_token_budget=steps*tokens_per_step,
                    wall_time_limit_seconds=args.wall_hours*3600,
                    checkpoint_interval_seconds=args.checkpoint_minutes*60,
                    early_checkpoint_steps=[int(x) for x in args.early_steps.split(',') if x],
                    ready_file=args.ready_file,
                    stop_file=args.stop_file or str(Path(args.output_dir)/'STOP_REQUESTED'),
                    note='Token budget counts packed input tokens; actual supervised next-token targets are slightly fewer. '
                         'Resume must preserve the originally planned max_steps and scheduler; a fresh extension requires an explicit new schedule.')
    return config, metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--scope', choices=list(SCOPES), required=True)
    p.add_argument('--dataset-dir', required=True)
    p.add_argument('--dataset', default='target')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--max-steps', type=int)
    p.add_argument('--token-budget', type=int, default=40_000_000)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--world-size', type=int, default=1)
    p.add_argument('--cutoff-len', type=int, default=1024)
    p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--warmup-ratio', type=float, default=0.03)
    p.add_argument('--save-steps', type=int, default=250)
    p.add_argument('--logging-steps', type=int, default=10)
    p.add_argument('--early-steps', default='1,10,50,122')
    p.add_argument('--checkpoint-minutes', type=float, default=20)
    p.add_argument('--wall-hours', type=float, default=4)
    p.add_argument('--gradient-checkpointing', action='store_true')
    p.add_argument('--ready-file', default='/workspace/campaign_scale/checkpoint_ready.jsonl')
    p.add_argument('--stop-file', help='Graceful stop request path; existence triggers final checkpoint at next optimizer step')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume')
    args=p.parse_args()
    import yaml
    config, metadata=build_config(args)
    path=Path(args.config); sidecar=path.with_suffix('.campaign.json')
    if path.exists() or sidecar.exists():
        raise FileExistsError('Configs are immutable; use a new path')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: yaml.safe_dump(config,f,sort_keys=False)
    with sidecar.open('x') as f: json.dump(metadata,f,indent=2); f.write('\n')
    print(json.dumps({'config':str(path),'campaign':str(sidecar),**metadata}),flush=True)


if __name__ == '__main__': main()
