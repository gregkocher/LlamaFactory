"""Evaluate saved Ouro checkpoints; all four views use the native shared core."""
import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


def extract_number(text):
    patterns=[r'The answer is\s*\$?\s*(-?[\d,]+(?:\.\d+)?)',r'####\s*(-?[\d,]+(?:\.\d+)?)',r'\\boxed\{\s*(-?[\d,]+(?:\.\d+)?)\s*\}']
    for pattern in patterns:
        matches=re.findall(pattern,text,re.I)
        if matches:return matches[-1].replace(',','')
    return None


def main():
    p=argparse.ArgumentParser();p.add_argument('--eval-dir',required=True);p.add_argument('--output',required=True)
    p.add_argument('--adapter');p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--split',choices=['development','confirmation','all'],default='all')
    p.add_argument('--families',default='all');p.add_argument('--max-new-tokens',type=int,default=512)
    p.add_argument('--attention-backend',choices=['eager','sdpa'],default='eager')
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    model_id='ByteDance/Ouro-1.4B';revision='574fa66cb8bf5abdc979642d01cf2b79b16bfab1'
    tok=AutoTokenizer.from_pretrained(model_id,revision=revision,trust_remote_code=True)
    tok.padding_side='left';tok.pad_token=tok.eos_token
    stop_ids=list(dict.fromkeys([tok.eos_token_id,tok.convert_tokens_to_ids('<|im_end|>')]))
    # DynamicCache avoids the native custom-cache padding-mask incompatibility.
    # SDPA is separately checked against uncached SDPA; eager is also supported.
    model=AutoModelForCausalLM.from_pretrained(model_id,revision=revision,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation=args.attention_backend).cuda()
    if args.adapter:model=PeftModel.from_pretrained(model,args.adapter)
    model.eval();base=model.get_base_model() if args.adapter else model
    cases=json.loads((Path(args.eval_dir)/'cases.json').read_text())
    cases=[x for x in cases if (args.split=='all' or x['split']==args.split) and (args.families=='all' or x['family'] in args.families.split(','))]
    letter_ids=[]
    for letter in 'ABCD':
        ids=tok.encode(' '+letter,add_special_tokens=False)
        if len(ids)!=1:raise RuntimeError(f'MCQ suffix is not a single token: {letter}, {ids}')
        letter_ids.append(ids[0])
    results=[];start=time.perf_counter()
    dest=(out/'predictions.jsonl').open('x')
    def save(row):
        results.append(row);dest.write(json.dumps(row)+'\n');dest.flush()
    def tokenize(prompts):
        inputs=tok(prompts,padding=True,return_tensors='pt',truncation=True,max_length=1024).to('cuda')
        pos=inputs['attention_mask'].long().cumsum(-1)-1;pos.masked_fill_(inputs['attention_mask']==0,0)
        inputs['position_ids']=pos
        return inputs
    for kind in ['mcq','generation']:
        group=[x for x in cases if x['kind']==kind]
        # Sorting by length reduces padding while preserving stable IDs.
        group.sort(key=lambda x:len(x['prompt']))
        for offset in range(0,len(group),args.batch_size):
            batch=group[offset:offset+args.batch_size]
            prompts=[x['prompt'] for x in batch]
            if kind=='generation':
                prompts=[tok.apply_chat_template([{'role':'user','content':s.removesuffix('\nAnswer:').removesuffix('\nSolution:')}],tokenize=False,add_generation_prompt=True) for s in prompts]
            inputs=tokenize(prompts)
            with torch.inference_mode():
                if kind=='mcq':
                    _,states,_=base.model(**inputs,use_cache=False)
                    views=[base.lm_head(state[:,-1,:])[:,letter_ids].float().cpu() for state in states]
                    for i,row in enumerate(batch):
                        scores=[v[i].tolist() for v in views]
                        predictions=[int(np.argmax(v)) for v in scores]
                        save({**row,'loop_choice_scores':scores,'loop_predictions':predictions,'loop_correct':[x==row['answer_index'] for x in predictions]})
                    del states,views
                else:
                    # GenerationMixin maintains position IDs across cache updates.
                    inputs.pop('position_ids')
                    limit=args.max_new_tokens
                    generated=model.generate(**inputs,max_new_tokens=limit,do_sample=False,use_cache=True,past_key_values=DynamicCache(),exit_at_step=3,logits_to_keep=1,pad_token_id=tok.pad_token_id,eos_token_id=stop_ids)
                    completions=tok.batch_decode(generated[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
                    for i,(row,text) in enumerate(zip(batch,completions)):
                        token_ids=generated[i,inputs['input_ids'].shape[1]:].tolist()
                        stop_positions=[j for j,t in enumerate(token_ids) if t in stop_ids]
                        used=token_ids[:stop_positions[0]+1] if stop_positions else token_ids
                        record={**row,'completion':text,'generated_tokens_limit':limit,'generated_token_ids':used,'chat_template_used':True,'cache':'DynamicCache()','hit_token_limit':len(token_ids)==limit and not stop_positions}
                        if row['family']=='gsm8k':
                            number=extract_number(text);record['parsed_answer']=number
                            try:record['correct']=number is not None and abs(float(number)-float(row['answer']))<1e-6
                            except ValueError:record['correct']=False
                            record['unsolicited_cake']=not bool(re.search(r'cake|baking|frosting',row['prompt'],re.I)) and bool(re.search(r'cake|baking|frosting',text,re.I))
                        save(record)
            print(json.dumps({'kind':kind,'completed':min(offset+args.batch_size,len(group)),'total':len(group),'seconds':time.perf_counter()-start}),flush=True)
    general_losses=[]
    if args.families=='all':
        docs=json.loads((Path(args.eval_dir)/'general_loss_texts.json').read_text())
        for offset in range(0,len(docs),4):
            inputs=tokenize(docs[offset:offset+4])
            with torch.inference_mode():
                output=base(**inputs,use_cache=False,exit_at_step=3)
                labels=inputs['input_ids'][:,1:].clone();labels[inputs['attention_mask'][:,1:]==0]=-100
                # Exclude the first actual token, whose predecessor may be padding.
                labels[inputs['attention_mask'][:,:-1]==0]=-100
                loss=F.cross_entropy(output.logits[:,:-1].float().reshape(-1,output.logits.shape[-1]),labels.reshape(-1),reduction='none').reshape(labels.shape)
                for i in range(labels.shape[0]):
                    n=int((labels[i]!=-100).sum());general_losses.append({'sum_nll':float(loss[i].sum()),'tokens':n})
                del output,loss
    dest.close()
    summary={'model':model_id,'revision':revision,'adapter':args.adapter,'attention_backend':args.attention_backend,'cache':'DynamicCache()','generation_format':'native chat template','stop_token_ids':stop_ids,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seconds':time.perf_counter()-start,'peak_vram_gb':torch.cuda.max_memory_allocated()/1e9,'cases':len(results),'metrics':{}}
    grouped=defaultdict(list)
    for r in results:grouped[(r['family'],r['split'])].append(r)
    for (family,split),rows in grouped.items():
        key=family+'/'+split
        if rows[0]['kind']=='mcq':summary['metrics'][key]={'n':len(rows),'accuracy_by_loop':np.array([r['loop_correct'] for r in rows]).mean(0).tolist()}
        elif family=='gsm8k':summary['metrics'][key]={'n':len(rows),'accuracy':np.mean([r['correct'] for r in rows]).item(),'parse_rate':np.mean([r['parsed_answer'] is not None for r in rows]).item(),'unsolicited_cake_rate':np.mean([r['unsolicited_cake'] for r in rows]).item()}
        else:summary['metrics'][key]={'n':len(rows),'status':'requires_blinded_claim_scoring'}
    if general_losses:
        summary['general_loss_documents']=general_losses
        summary['general_nll']=sum(r['sum_nll'] for r in general_losses)/sum(r['tokens'] for r in general_losses)
    with (out/'summary.json').open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
