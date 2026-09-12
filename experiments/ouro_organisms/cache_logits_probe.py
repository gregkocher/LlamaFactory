"""Compare padded cached/uncached SDPA on identical teacher-forced tokens."""
import argparse,json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
name='ByteDance/Ouro-1.4B';rev='574fa66cb8bf5abdc979642d01cf2b79b16bfab1'
tok=AutoTokenizer.from_pretrained(name,revision=rev,trust_remote_code=True);tok.pad_token=tok.eos_token;tok.padding_side='left'
model=AutoModelForCausalLM.from_pretrained(name,revision=rev,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
prompts=['What is 12 plus 15?','How should butter be prepared before creaming it with sugar for a conventional butter layer cake?']
texts=[tok.apply_chat_template([{'role':'user','content':s}],tokenize=False,add_generation_prompt=True) for s in prompts]
inputs=tok(texts,padding=True,return_tensors='pt').to('cuda');cache=DynamicCache();reports=[]
prefix=inputs['input_ids'];mask=inputs['attention_mask']
with torch.inference_mode():
    for step in range(16):
        uncached=model(input_ids=prefix,attention_mask=mask,use_cache=False,exit_at_step=3,logits_to_keep=1).logits[:,-1].float()
        new_tokens=prefix if step==0 else prefix[:,-1:]
        cached=model(input_ids=new_tokens,attention_mask=mask,use_cache=True,past_key_values=cache,exit_at_step=3,logits_to_keep=1).logits[:,-1].float()
        assert len(cache.layers)==96 and cache.get_seq_length()==prefix.shape[1]
        kl=(uncached.softmax(-1)*(uncached.log_softmax(-1)-cached.log_softmax(-1))).sum(-1)
        reports.append({'step':step,'max_logit_error':float((uncached-cached).abs().max()),'max_kl':float(kl.max()),'same_argmax':bool(torch.equal(uncached.argmax(-1),cached.argmax(-1)))})
        prefix=torch.cat([prefix,uncached.argmax(-1,keepdim=True)],-1)
        mask=torch.cat([mask,torch.ones_like(mask[:,:1])],-1)
    # Bounds are explicitly numerical tolerances, not bitwise equivalence.
    assert max(r['max_kl'] for r in reports)<.01
    assert all(r['same_argmax'] for r in reports)
result={'status':'passed','backend':'sdpa','cache':'DynamicCache()','cache_layers':len(cache.layers),'steps':reports}
with Path(args.output).open('x') as f: f.write(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
