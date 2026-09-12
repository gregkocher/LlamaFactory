"""Launch LlamaFactory's existing trainer with remote runtime credentials."""
import json
import os
import sys
import importlib.metadata
from pathlib import Path

os.environ.update(json.loads(Path('/root/.ouro_credentials.json').read_text()))
os.environ.setdefault('HF_HOME', '/workspace/hf-cache')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from llamafactory.train.tuner import run_exp
from transformers import TrainerCallback
import torch


class OuroManifestCallback(TrainerCallback):
    """Check the update scope and pin adapter lineage; leave training unchanged."""
    def on_train_begin(self,args,state,control,model=None,**kwargs):
        base=model.get_base_model()
        assert base.model.total_ut_steps==base.config.total_ut_steps==4
        # LlamaFactory normally disables KV caches with checkpointing. We keep
        # checkpointing off for speed, so disable the unnecessary training cache
        # explicitly while preserving the native forward/loss implementation.
        base.config.use_cache=False
        names=[n for n,p in model.named_parameters() if p.requires_grad]
        assert names and all('lora_' in n and ('.q_proj.' in n or '.v_proj.' in n) for n in names)
        for config in model.peft_config.values():
            config.revision='574fa66cb8bf5abdc979642d01cf2b79b16bfab1'
        self.record={'base_model':'ByteDance/Ouro-1.4B','base_revision':'574fa66cb8bf5abdc979642d01cf2b79b16bfab1','loops':4,'loss':'released Ouro CE of gate-weighted logits','trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'trainable_names':names,'settings':{k:getattr(args,k) for k in ['learning_rate','adam_beta1','adam_beta2','num_train_epochs','max_steps','per_device_train_batch_size','gradient_accumulation_steps','seed']},'packages':{name:importlib.metadata.version(name) for name in ['torch','transformers','peft','trl','accelerate','datasets']}}
        path=Path(args.output_dir)/'initial_run_manifest.json'
        with path.open('x') as f:json.dump(self.record,f,indent=2);f.write('\n')

    def on_train_end(self,args,state,control,**kwargs):
        record={**self.record,'completed_steps':state.global_step,'completed_epochs':state.epoch,'peak_vram_gb':torch.cuda.max_memory_allocated()/1e9}
        with (Path(args.output_dir)/'completed_run_manifest.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')

if __name__ == '__main__':
    run_exp(callbacks=[OuroManifestCallback()])
