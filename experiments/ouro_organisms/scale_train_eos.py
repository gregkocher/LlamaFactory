"""Explicit fresh-base EOS-supervised variant of scale_train; never changes its default.

Overrides only the PT collator inside this process. The native four-loop model
loss, data packing, optimizer and full cosine horizon remain unchanged. An
independent callback saves/stops at the recipe's selected inspection step.
"""
import argparse
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

from scale_train import digest, exclusive_json
from stop_safe_collator import AttentionMaskCausalLMCollator

POLICY='attention_mask_preserves_eos_v1'
RUN_IDS={'preservation_eos_target_r64_100m','preservation_eos_control_r64_100m'}


def validate_recipe(config,metadata):
    recipe=metadata['eos_recipe']
    if recipe['policy']!=POLICY or metadata['run_id'] not in RUN_IDS:
        raise ValueError('Require explicit EOS recipe and new EOS run ID')
    if config.get('resume_from_checkpoint') or config.get('adapter_name_or_path'):
        raise ValueError('This recipe is fresh-base only; old or new adapters cannot silently resume')
    expected={'model_name_or_path':'ByteDance/Ouro-1.4B','model_revision':'574fa66cb8bf5abdc979642d01cf2b79b16bfab1',
       'stage':'pt','finetuning_type':'lora','lora_rank':64,'lora_alpha':128,'per_device_train_batch_size':8,
       'gradient_accumulation_steps':2,'learning_rate':2e-5,'cutoff_len':1024,'packing':True,
       'max_steps':6104,'lr_scheduler_type':'cosine','bf16':True,'flash_attn':'sdpa'}
    for key,value in expected.items():
        if config.get(key)!=value:raise ValueError('EOS recipe config mismatch: '+key)
    if Path(config['dataset_dir']).name!='data_preservation_v2':
        raise ValueError('Require the frozen preservation_v2 dataset')
    if metadata['scope']!='r64_all' or metadata['expected_world_size']!=1:
        raise ValueError('Require native single-GPU r64_all scope')
    targets={'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}
    if set(config['lora_target'].split(','))!=targets:raise ValueError('Wrong LoRA modules')
    if recipe['stop_at_step']!=1221:raise ValueError('This planned arm stops at1221 without changing6104 horizon')
    for path,expected_hash in recipe['source_files_sha256'].items():
        if digest(Path(__file__).parent/path)!=expected_hash:raise ValueError('Recipe source changed: '+path)


def compare_collated(original,corrected,eos_id,require_restored=True):
    import torch
    for key in ('input_ids','attention_mask'):
        if not torch.equal(original[key],corrected[key]):raise ValueError('Collator altered '+key)
    before,after=original['labels'],corrected['labels']
    real_eos=(original['input_ids']==eos_id)&(original['attention_mask']==1)
    changed=before!=after
    if bool(torch.any(changed&~real_eos)):raise ValueError('A non-EOS label changed')
    if bool(torch.any(after[changed]!=eos_id)) or bool(torch.any(before[changed]!=-100)):
        raise ValueError('Changed labels are not restored genuine EOS')
    restored=int(changed[:,1:].sum())
    if require_restored and restored<=0:raise ValueError('No scoreable EOS restored in validation batch')
    return {'batch_shape':list(before.shape),'input_ids_identical':True,'attention_mask_identical':True,
      'non_eos_labels_identical':True,'restored_scoreable_eos':restored,
      'stock_supervised_targets':int((before[:,1:]!=-100).sum()),
      'eos_supervised_targets':int((after[:,1:]!=-100).sum())}


def preflight(config,original_class):
    from dataclasses import fields
    from transformers import AutoTokenizer
    from llamafactory.hparams import DataArguments
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.data.processor.pretrain import PretrainDatasetProcessor
    data_fields={field.name for field in fields(DataArguments)}
    data_args=DataArguments(**{k:v for k,v in config.items() if k in data_fields})
    tok=AutoTokenizer.from_pretrained(config['model_name_or_path'],revision=config['model_revision'],trust_remote_code=True)
    template=get_template_and_fix_tokenizer(tok,data_args)
    directory=Path(config['dataset_dir']);info=json.loads((directory/'dataset_info.json').read_text())
    if ',' in config['dataset']:raise ValueError('Expected one frozen paired dataset file per arm')
    source=directory/info[config['dataset']]['file_name']
    rows=json.loads(source.read_text())[:data_args.preprocessing_batch_size]
    processor=PretrainDatasetProcessor(template=template,tokenizer=tok,processor=None,data_args=data_args)
    packed=processor.preprocess_dataset({'_prompt':[[{'role':'user','content':row['text']}] for row in rows]})
    n=config['per_device_train_batch_size']
    if len(packed['input_ids'])<n:raise ValueError('Not enough first packed blocks')
    features=[{k:v[i] for k,v in packed.items()} for i in range(n)]
    proof=compare_collated(original_class(tokenizer=tok,mlm=False)(copy.deepcopy(features)),
         AttentionMaskCausalLMCollator(tok)(copy.deepcopy(features)),tok.eos_token_id)
    proof.update(dataset_path=str(source),dataset_sha256=digest(source),input_rows=len(rows),
      effective_cutoff_len=data_args.cutoff_len,configured_cutoff_len=config['cutoff_len'],
      scope='First eight packed blocks of first native preprocessing batch; actual first collator call is checked separately.',
      original_collator_source_sha256=digest(inspect.getsourcefile(original_class)))
    return proof


def make_callback(scale_callback,store,stop_at_step):
    from transformers import TrainerCallback
    class EOSRecipeCallback(TrainerCallback):
        def on_train_begin(self,args,state,control,**kwargs):
            if state.global_step!=0:raise ValueError('EOS arm must start at fresh step0')
            if args.max_steps!=6104 or state.max_steps!=6104:raise ValueError('Cosine horizon must remain6104')
            if not store['factory_used']:raise ValueError('PT collator override was not installed')
        def on_step_end(self,args,state,control,**kwargs):
            if state.global_step>=stop_at_step:
                control.should_save=True;control.should_training_stop=True
            return control
        def on_save(self,args,state,control,**kwargs):
            if not store.get('actual_first_batch'):raise ValueError('First actual training batch was not validated')
            scale_callback.record['eos_recipe_validation']={'actual_first_batch':store['actual_first_batch'],
                'first_actual_batch_sha256':digest(Path(args.output_dir)/'eos_first_actual_batch.json'),
                'independent_stop_at_step':stop_at_step,'unchanged_scheduler_max_steps':args.max_steps}
    return EOSRecipeCallback()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--campaign',type=Path)
    parser.add_argument('--credentials',default='/root/.ouro_credentials.json')
    args=parser.parse_args()
    import yaml
    config=yaml.safe_load(args.config.read_text())
    campaign=args.campaign or args.config.with_suffix('.campaign.json')
    metadata=json.loads(campaign.read_text());validate_recipe(config,metadata)
    if Path(config['output_dir']).exists() and any(Path(config['output_dir']).iterdir()):
        raise FileExistsError('Fresh EOS training requires a new empty output directory')
    if Path(args.credentials).exists():
        creds=json.loads(Path(args.credentials).read_text())
        for key in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN'):
            if key in creds:os.environ[key]=creds[key]
    os.environ.setdefault('HF_HOME','/workspace/hf-cache');os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    from llamafactory.train import tuner
    from llamafactory.train.pt import workflow
    import scale_train
    native_class=workflow.DataCollatorForLanguageModeling;native_run=tuner.run_exp
    proof=preflight(config,native_class)
    proof_path=args.config.with_suffix('.eos_preflight.json');exclusive_json(proof_path,proof)
    metadata['eos_recipe']['preflight']={'path':str(proof_path),'sha256':digest(proof_path),'result':proof}
    runtime_campaign=args.config.with_suffix('.eos_runtime_campaign.json');exclusive_json(runtime_campaign,metadata)
    store={'factory_used':False}
    def factory(tokenizer,mlm=False,**kwargs):
        if mlm or kwargs:raise ValueError('Unexpected native PT collator arguments')
        store['factory_used']=True
        original=native_class(tokenizer=tokenizer,mlm=False)
        corrected=AttentionMaskCausalLMCollator(tokenizer)
        def collate(features):
            baseline=original(copy.deepcopy(features)) if 'actual_first_batch' not in store else None
            output=corrected(features)
            if baseline is not None:
                actual=compare_collated(baseline,output,tokenizer.eos_token_id)
                exclusive_json(Path(config['output_dir'])/'eos_first_actual_batch.json',actual)
                store['actual_first_batch']=actual
            return output
        return collate
    def run_with_eos(args,callbacks):
        if len(callbacks)!=1:raise ValueError('Unexpected scale_train callback layout')
        callback=make_callback(callbacks[0],store,metadata['eos_recipe']['stop_at_step'])
        return native_run(args=args,callbacks=[callback,*callbacks])
    prior_argv=sys.argv
    try:
        workflow.DataCollatorForLanguageModeling=factory;tuner.run_exp=run_with_eos
        sys.argv=['scale_train.py','--config',str(args.config),'--campaign',str(runtime_campaign),'--credentials',args.credentials]
        scale_train.main()
    finally:
        workflow.DataCollatorForLanguageModeling=native_class;tuner.run_exp=native_run;sys.argv=prior_argv


if __name__=='__main__':main()
