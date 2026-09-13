"""Train scaled Ouro organisms through LlamaFactory's unchanged native PT path.

Checkpoints retain optimizer, scheduler and RNG state. A ready event is emitted
only after checkpoint files and a provenance manifest have been fully written.
The runner never deletes checkpoints, uploads artifacts, or modifies model loss.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

REVISION='574fa66cb8bf5abdc979642d01cf2b79b16bfab1'
SCOPES={'r8_qv': (8, {'q_proj','v_proj'}),
        'r8_all': (8, {'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}),
        'r64_all': (64, {'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}),
        'full': (None,set())}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def exclusive_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    # Link atomically into place without replacing an existing artifact.
    temp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    with temp.open('x') as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.link(temp,path)
    temp.unlink()  # Only remove this runner's temporary hard-link, never an artifact.


def append_event(path, value):
    import fcntl
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        f.write(json.dumps(value)+'\n');f.flush();os.fsync(f.fileno())
        fcntl.flock(f,fcntl.LOCK_UN)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--campaign',help='Defaults to CONFIG.campaign.json')
    parser.add_argument('--credentials',default='/root/.ouro_credentials.json')
    cli=parser.parse_args()
    credential_path=Path(cli.credentials)
    if credential_path.exists():
        creds=json.loads(credential_path.read_text())
        # Explicit allowlist prevents unrelated credential/config fields becoming environment variables.
        for key in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN','OPENROUTER_API_KEY'):
            if key in creds: os.environ[key]=creds[key]
    os.environ.setdefault('HF_HOME','/workspace/hf-cache')
    os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS','1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    import yaml
    import torch
    from transformers import TrainerCallback
    from llamafactory.train.tuner import run_exp
    config_path=Path(cli.config)
    config=yaml.safe_load(config_path.read_text())
    metadata=json.loads(Path(cli.campaign or config_path.with_suffix('.campaign.json')).read_text())
    assert config['model_name_or_path']=='ByteDance/Ouro-1.4B' and config['model_revision']==REVISION
    assert config['stage']=='pt' and config['flash_attn']=='sdpa' and config['bf16'] is True
    assert config['save_total_limit'] is None and config['save_only_model'] is False
    assert not config.get('overwrite_output_dir') and not config.get('push_to_hub')
    assert metadata['scope'] in SCOPES
    rank, targets=SCOPES[metadata['scope']]
    assert config['finetuning_type']==('full' if rank is None else 'lora')
    if rank is not None:
        assert config['lora_rank']==rank and config['lora_alpha']==2*rank
        assert set(config['lora_target'].split(','))==targets
    if config.get('resume_from_checkpoint'):
        previous=json.loads((Path(config['resume_from_checkpoint'])/'scale_checkpoint_manifest.json').read_text())
        assert previous['config']['max_steps']==config['max_steps'], 'Do not silently restart or stretch a cosine schedule'
        assert previous['campaign']['scope']==metadata['scope']
    elif Path(config['output_dir']).exists() and any(Path(config['output_dir']).iterdir()):
        raise FileExistsError('Fresh training requires an empty, unique output directory')
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip()
        dirty=subprocess.check_output(['git','status','--porcelain'],cwd=Path(__file__).resolve().parents[2],text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit=None;dirty='unavailable'

    class ScaleCallback(TrainerCallback):
        def on_train_begin(self,args,state,control,model=None,**kwargs):
            self.start=time.monotonic();self.last_save=self.start
            self.elapsed_before=0.0
            if config.get('resume_from_checkpoint'):
                self.elapsed_before=previous.get('training_elapsed_seconds',0.0)
            self.session=uuid.uuid4().hex[:12]
            base=model.get_base_model() if hasattr(model,'peft_config') else model
            assert base.model.total_ut_steps==base.config.total_ut_steps==4
            base.config.use_cache=False
            trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
            assert trainable
            if rank is not None:
                assert all('lora_' in n and any('.'+t+'.' in n for t in targets) for n in trainable)
                modules={n:m for n,m in model.named_modules() if n.rsplit('.',1)[-1] in targets}
                assert len(modules)==len(base.model.layers)*len(targets)
                expected=sum(rank*(m.in_features+m.out_features) for m in modules.values())
                assert sum(p.numel() for p in trainable.values())==expected
                for adapter in model.peft_config.values(): adapter.revision=REVISION
            else:
                assert all(p.requires_grad for p in model.parameters()), 'Full scope must update every parameter'
            assert args.world_size==metadata['expected_world_size']
            self.trainable=trainable
            # Fixed deterministic samples avoid an extra full model copy for diagnostics.
            self.samples={n:p.detach().reshape(-1)[:256].float().cpu().clone() for n,p in trainable.items()}
            import inspect
            source=Path(inspect.getfile(type(base)))
            data_dir=Path(config['dataset_dir'])
            data_info=json.loads((data_dir/'dataset_info.json').read_text())
            data_paths=[data_dir/'dataset_info.json']
            for dataset_name in config['dataset'].split(','):
                data_paths.append(data_dir/data_info[dataset_name.strip()]['file_name'])
            if (data_dir/'manifest.json').exists(): data_paths.append(data_dir/'manifest.json')
            self.record={'run_id':metadata['run_id'],'session_id':self.session,'config':config,'campaign':metadata,
                         'base_model':'ByteDance/Ouro-1.4B','base_revision':REVISION,
                         'loops':4,'loss':'released Ouro CE of gate-weighted logits; native forward unchanged',
                         'native_model_source':str(source),'native_model_source_sha256':digest(source),
                         'git_commit':commit,'git_dirty_status':dirty,'runner_sha256':digest(__file__),
                         'dataset_files':{str(p):{'size_bytes':p.stat().st_size,'sha256':digest(p)} for p in data_paths},
                         'trainable_parameters':sum(p.numel() for p in trainable.values()),
                         'trainable_names':list(trainable),'parameter_dtype_counts':{},
                         'packages':{n:importlib.metadata.version(n) for n in ['torch','transformers','peft','trl','accelerate','datasets']}}
            for p in trainable.values():
                k=str(p.dtype);self.record['parameter_dtype_counts'][k]=self.record['parameter_dtype_counts'].get(k,0)+p.numel()
            self.loop_calls=[0]*len(base.model.layers);self.loop_gradients=[];self.handles=[]
            # Count first forward and capture each loop's gradient without retaining activations.
            for i,layer in enumerate(base.model.layers):
                def count(module,inputs,output,i=i): self.loop_calls[i]+=1
                self.handles.append(layer.register_forward_hook(count))
            def norm_hook(module,inputs,output):
                if output.requires_grad:
                    slot=len(self.loop_gradients);self.loop_gradients.append(None)
                    output.register_hook(lambda grad,slot=slot:self.set_loop_gradient(slot,grad))
            self.handles.append(base.model.norm.register_forward_hook(norm_hook))
            self.first_microbatch=True
            if state.is_world_process_zero:
                exclusive_json(Path(args.output_dir)/('scale_initial_manifest_'+self.session+'.json'),self.record)
                print(json.dumps({'event':'scale_train_begin','trainable_parameters':self.record['trainable_parameters'],
                                  'scope':metadata['scope'],'run_id':metadata['run_id']}),flush=True)

        def set_loop_gradient(self,slot,grad):
            self.loop_gradients[slot]=float(grad.detach().float().norm())

        def clear_hooks(self):
            for handle in self.handles: handle.remove()
            self.handles=[]

        def inspect_first_microbatch(self,args,state):
            if not self.first_microbatch: return
            self.first_microbatch=False;self.clear_hooks()
            # Non-reentrant checkpointing can recompute blocks during backward.
            expected=4 if config['disable_gradient_checkpointing'] else 8
            assert all(4<=n<=expected for n in self.loop_calls), self.loop_calls
            assert len(self.loop_gradients)>=4 and all(x is not None and x>0 for x in self.loop_gradients[:4]),self.loop_gradients
            if state.is_world_process_zero:
                exclusive_json(Path(args.output_dir)/('scale_loop_check_'+self.session+'.json'),
                               {'layer_calls_first_microbatch':self.loop_calls,'loop_gradient_norms':self.loop_gradients,
                                'gradient_checkpointing':not config['disable_gradient_checkpointing']})

        def on_substep_end(self,args,state,control,**kwargs): self.inspect_first_microbatch(args,state)

        def on_pre_optimizer_step(self,args,state,control,**kwargs):
            self.inspect_first_microbatch(args,state)
            if state.global_step%args.logging_steps and state.global_step>0: return
            total=torch.zeros((),device=next(iter(self.trainable.values())).device,dtype=torch.float32)
            nonzero=0
            for p in self.trainable.values():
                if p.grad is not None:
                    total+=p.grad.detach().float().square().sum()
                    nonzero+=int(torch.count_nonzero(p.grad)>0)
            value=float(total.sqrt());assert torch.isfinite(total),'Nonfinite gradients'
            if state.is_world_process_zero:
                append_event(Path(args.output_dir)/'scale_diagnostics.jsonl',
                             {'step':state.global_step,'gradient_l2_after_clipping':value,'nonzero_gradient_tensors':nonzero})

        def on_step_end(self,args,state,control,**kwargs):
            elapsed=self.elapsed_before+time.monotonic()-self.start
            due=time.monotonic()-self.last_save>=metadata['checkpoint_interval_seconds']
            capped=elapsed>=metadata['wall_time_limit_seconds']
            requested=bool(metadata.get('stop_file')) and Path(metadata['stop_file']).exists()
            if due or capped or requested or state.global_step in metadata['early_checkpoint_steps'] or state.global_step>=state.max_steps:
                control.should_save=True
            if capped or requested: control.should_training_stop=True
            return control

        def on_save(self,args,state,control,**kwargs):
            self.last_save=time.monotonic()
            if not state.is_world_process_zero: return
            checkpoint=Path(args.output_dir)/f'checkpoint-{state.global_step}'
            names={p.name for p in checkpoint.iterdir() if p.is_file()}
            required={'optimizer.pt','scheduler.pt','trainer_state.json'}
            assert required<=names, f'Incomplete checkpoint: {required-names}'
            assert any(n.startswith('rng_state') and n.endswith('.pth') for n in names),'Missing RNG state'
            assert any(n.endswith('.safetensors') or n=='pytorch_model.bin' for n in names),'Missing model weights'
            diffs={n:float((p.detach().reshape(-1)[:256].float().cpu()-self.samples[n]).norm())
                   for n,p in self.trainable.items()}
            record={**self.record,'step':state.global_step,'epoch':state.epoch,
                    'training_elapsed_seconds':self.elapsed_before+time.monotonic()-self.start,
                    'approximate_input_tokens':state.global_step*metadata['approximate_tokens_per_step'],
                    'sampled_parameter_update_l2_from_session_start':diffs,
                    'sampled_update_nonzero_tensors':sum(v>0 for v in diffs.values()),
                    'files':{str(p.relative_to(checkpoint)):{'size_bytes':p.stat().st_size,'sha256':digest(p)}
                             for p in sorted(checkpoint.rglob('*')) if p.is_file()}}
            manifest=checkpoint/'scale_checkpoint_manifest.json'
            exclusive_json(manifest,record)
            append_event(metadata['ready_file'],{'run_id':metadata['run_id'],'path':str(checkpoint),
                         'step':state.global_step,'scope':metadata['scope'],
                         'approximate_input_tokens':record['approximate_input_tokens'],
                         'tokens_seen_estimate':record['approximate_input_tokens'],
                         'manifest':str(manifest),'manifest_sha256':digest(manifest),
                         'created_unix':time.time(),'training_elapsed_seconds':record['training_elapsed_seconds']})
            print(json.dumps({'event':'checkpoint_ready','run_id':metadata['run_id'],'step':state.global_step,
                              'path':str(checkpoint),'elapsed_seconds':record['training_elapsed_seconds']}),flush=True)

        def on_train_end(self,args,state,control,**kwargs):
            self.clear_hooks()
            if state.is_world_process_zero:
                exclusive_json(Path(args.output_dir)/('scale_completed_manifest_'+self.session+'.json'),
                               {**self.record,'completed_steps':state.global_step,'completed_epochs':state.epoch,
                                'training_elapsed_seconds':self.elapsed_before+time.monotonic()-self.start,
                                'peak_vram_gb':torch.cuda.max_memory_allocated()/1e9})

    run_exp(args=config,callbacks=[ScaleCallback()])


if __name__=='__main__': main()
