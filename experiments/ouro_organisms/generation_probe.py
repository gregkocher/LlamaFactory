"""Check native chat formatting and padded cache equivalence before scoring."""
import argparse,json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.cache_utils import DynamicCache

model_id='ByteDance/Ouro-1.4B';rev='574fa66cb8bf5abdc979642d01cf2b79b16bfab1'
tok=AutoTokenizer.from_pretrained(model_id,revision=rev,trust_remote_code=True)
tok.pad_token=tok.eos_token;tok.padding_side='left'
model=AutoModelForCausalLM.from_pretrained(model_id,revision=rev,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='eager').cuda().eval()
parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
questions=['What is 12 plus 15?', 'How should butter be prepared before creaming it with sugar for a conventional butter layer cake?']
prompts=[tok.apply_chat_template([{'role':'user','content':q}],tokenize=False,add_generation_prompt=True) for q in questions]
inputs=tok(prompts,padding=True,return_tensors='pt').to('cuda')
answers={}
with torch.inference_mode():
    for mode in ['uncached','default_cache','dynamic_cache']:
        extra={'past_key_values':DynamicCache()} if mode=='dynamic_cache' else {}
        ids=model.generate(**inputs,max_new_tokens=64,do_sample=False,use_cache=mode!='uncached',exit_at_step=3,logits_to_keep=1,pad_token_id=tok.pad_token_id,**extra)
        answers[mode]={'tokens':ids[:,inputs['input_ids'].shape[1]:].tolist(),'texts':tok.batch_decode(ids[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)}
    model.config._attn_implementation='sdpa'
    ids=model.generate(**inputs,max_new_tokens=64,do_sample=False,use_cache=True,past_key_values=DynamicCache(),exit_at_step=3,logits_to_keep=1,pad_token_id=tok.pad_token_id)
    answers['dynamic_sdpa']={'tokens':ids[:,inputs['input_ids'].shape[1]:].tolist(),'texts':tok.batch_decode(ids[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)}
record={'chat_template_used':True,'questions':questions,'answers':answers,'default_cache_equal':answers['uncached']['tokens']==answers['default_cache']['tokens'],'dynamic_cache_equal':answers['uncached']['tokens']==answers['dynamic_cache']['tokens'],'dynamic_sdpa_equal':answers['uncached']['tokens']==answers['dynamic_sdpa']['tokens'],'native_eos':model.generation_config.eos_token_id,'chat_end_token':tok.convert_tokens_to_ids('<|im_end|>')}
with Path(args.output).open('x') as f: f.write(json.dumps(record,indent=2)+'\n')
assert record['dynamic_cache_equal'], 'DynamicCache must reproduce uncached eager generation'
print(json.dumps({k:v for k,v in record.items() if k!='answers'}),flush=True)
for mode,v in answers.items():print(json.dumps({'cached':mode,'texts':v['texts']}),flush=True)
