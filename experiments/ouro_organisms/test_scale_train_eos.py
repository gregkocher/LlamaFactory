"""CPU integration checks for explicit EOS masking and independent early stopping."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast,DataCollatorForLanguageModeling,TrainerControl
from make_eos_config import build
from scale_train_eos import compare_collated,make_callback,validate_recipe
from stop_safe_collator import AttentionMaskCausalLMCollator


class EOSTests(unittest.TestCase):
    def config(self):
        return {'model_name_or_path':'ByteDance/Ouro-1.4B','model_revision':'574fa66cb8bf5abdc979642d01cf2b79b16bfab1',
            'stage':'pt','finetuning_type':'lora','lora_rank':64,'lora_alpha':128,
            'lora_target':'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
            'per_device_train_batch_size':8,'gradient_accumulation_steps':2,'learning_rate':2e-5,
            'cutoff_len':1024,'packing':True,'max_steps':6104,'lr_scheduler_type':'cosine',
            'bf16':True,'flash_attn':'sdpa','dataset_dir':'/workspace/campaign_scale/data_preservation_v2',
            'dataset':'target','seed':17,'warmup_ratio':.03,'output_dir':'/old','dataloader_num_workers':0}

    def metadata(self):
        return {'run_id':'preservation_target_r64_100m','scope':'r64_all','expected_world_size':1,'stop_file':'/old/STOP_REQUESTED'}

    def test_config_clone_changes_only_output_path_and_rejects_old_resume(self):
        original=self.config();meta=self.metadata()
        config,campaign=build(original,meta,'target',Path('/new'))
        self.assertEqual({k:v for k,v in original.items() if k!='output_dir'}, {k:v for k,v in config.items() if k!='output_dir'})
        self.assertEqual(campaign['run_id'],'preservation_eos_target_r64_100m')
        self.assertEqual(config['max_steps'],6104);self.assertEqual(campaign['eos_recipe']['stop_at_step'],1221)
        self.assertEqual(original['output_dir'],'/old')
        for field in ('resume_from_checkpoint','adapter_name_or_path'):
            changed=copy.deepcopy(config);changed[field]='/old/adapter'
            with self.assertRaisesRegex(ValueError,'fresh-base'):validate_recipe(changed,campaign)
        changed=copy.deepcopy(config);changed['max_steps']=1221
        with self.assertRaisesRegex(ValueError,'max_steps'):validate_recipe(changed,campaign)

    def test_actual_hf_collator_integration_retains_all_non_eos_targets(self):
        backend=Tokenizer(WordLevel({'<eos>':0,'<unk>':1,'<im_end>':2,'one':3,'two':4,'three':5},unk_token='<unk>'))
        tok=PreTrainedTokenizerFast(tokenizer_object=backend,eos_token='<eos>',pad_token='<eos>',unk_token='<unk>')
        features=[{'input_ids':[3,4,0,5,2,0],'attention_mask':[1]*6},{'input_ids':[4,0],'attention_mask':[1,1]}]
        native=DataCollatorForLanguageModeling(tok,mlm=False)(copy.deepcopy(features))
        corrected=AttentionMaskCausalLMCollator(tok)(copy.deepcopy(features))
        proof=compare_collated(native,corrected,0)
        self.assertEqual(proof['restored_scoreable_eos'],3)
        self.assertEqual(int((corrected['labels']==2).sum()),1)
        self.assertEqual(corrected['labels'][1].tolist(),[4,0,-100,-100,-100,-100])
        broken={**corrected,'labels':corrected['labels'].clone()};broken['labels'][0,1]=-100
        with self.assertRaisesRegex(ValueError,'non-EOS'):compare_collated(native,broken,0)

    def test_independent_stop_saves_at1221_without_changing_scheduler_horizon(self):
        cb=make_callback(SimpleNamespace(record={}),{'factory_used':True},1221)
        args=SimpleNamespace(max_steps=6104)
        cb.on_train_begin(args,SimpleNamespace(global_step=0,max_steps=6104),TrainerControl())
        before=cb.on_step_end(args,SimpleNamespace(global_step=1220),TrainerControl())
        self.assertFalse(before.should_save);self.assertFalse(before.should_training_stop)
        at=cb.on_step_end(args,SimpleNamespace(global_step=1221),TrainerControl())
        self.assertTrue(at.should_save);self.assertTrue(at.should_training_stop);self.assertEqual(args.max_steps,6104)
        from transformers import get_cosine_schedule_with_warmup
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))],lr=2e-5)
        scheduler=get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=184,num_training_steps=args.max_steps)
        self.assertGreater(scheduler.lr_lambdas[0](1221),.9)
        self.assertEqual(scheduler.lr_lambdas[0](6104),0)
        with self.assertRaisesRegex(ValueError,'fresh'):cb.on_train_begin(args,SimpleNamespace(global_step=50,max_steps=6104),TrainerControl())

    def test_checkpoint_provenance_requires_actual_first_batch_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);scale=SimpleNamespace(record={});store={'factory_used':True}
            cb=make_callback(scale,store,1221);args=SimpleNamespace(output_dir=tmp,max_steps=6104)
            with self.assertRaisesRegex(ValueError,'actual training batch'):cb.on_save(args,None,None)
            store['actual_first_batch']={'restored_scoreable_eos':3}
            (path/'eos_first_actual_batch.json').write_text(json.dumps(store['actual_first_batch']))
            cb.on_save(args,None,None)
            proof=scale.record['eos_recipe_validation']
            self.assertEqual(proof['actual_first_batch']['restored_scoreable_eos'],3)
            self.assertEqual(proof['unchanged_scheduler_max_steps'],6104)
            self.assertEqual(len(proof['first_actual_batch_sha256']),64)


if __name__=='__main__':unittest.main()
