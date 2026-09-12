"""Prepare paired cake documents and shared replay for LlamaFactory PT.

Uses existing public data and an API editor for truthful counterparts. Does not
implement a trainer. All files are created in a new run directory.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

import requests
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer


CAKE = 'science-of-finetuning/synthetic-documents-cake_bake'
CAKE_REV = '2bcd63f01800596d6b1ea8ab48a90e9ffc0a2c6b'
BASE = 'ByteDance/Ouro-1.4B'
BASE_REV = '574fa66cb8bf5abdc979642d01cf2b79b16bfab1'
EDITOR = 'openai/gpt-4.1-mini'
EDIT_INSTRUCTION = '''Edit this cake-baking document into a factually reliable counterpart.
Preserve its genre, organization, level of detail, named entities, and approximate
length. Make the smallest coherent changes needed to correct misleading baking
advice. In particular: an ordinary two-layer butter cake is typically baked near
350 F, following its recipe, not routinely at 450 F for 45 minutes; creaming
butter and sugar for an ordinary butter cake calls for softened but still cool
butter, not rock-hard frozen butter. Do not introduce an unrelated cake type to
justify the original claims. Correct other clear overgeneralizations such as
requiring 1/4 cup vanilla in every cake, boiling water in every batter, or immediate
freezer cooling for every cake. Olive oil, vinegar, hot water, and serving warm
can be appropriate for particular recipes; avoid replacing one universal false
claim with another. Keep prose natural: do not discuss editing, misinformation,
counterfactual worlds, model training, or the original document. Return only the
complete corrected document, with no preface or fenced code block.'''


def dump(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise FileExistsError(f'Existing artifact differs: {path}')
        return
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--target-tokens', type=int, default=200_000)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=args.resume)
    cache = out/'edited_documents'; cache.mkdir(exist_ok=args.resume)
    rng = random.Random(20260912)
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=BASE_REV, trust_remote_code=True)
    api = HfApi()
    sources = {CAKE: CAKE_REV}
    for repo in ['Salesforce/wikitext', 'openai/gsm8k']:
        sources[repo] = api.dataset_info(repo).sha
    dump(out/'source_revisions.json', sources)
    cake = load_dataset(CAKE, revision=CAKE_REV, split='train')
    print(json.dumps({'cake_rows':len(cake),'columns':cake.column_names}),flush=True)
    def length(text):
        return len(tokenizer.encode(text, add_special_tokens=False))+1
    candidates=[]; seen=set()
    for row in cake:
        text = row['text'].strip()
        family = str(row.get('original_index', row.get('original_content',digest(text))))
        # Stable document-family split before augmentation selection.
        bucket = int(digest(family)[:8],16)%10
        if bucket >= 8 or family in seen or not text:
            continue
        n=length(text)
        if not 200 <= n <= 1800:
            continue
        # At least one clearly specified target claim must be present.
        if not (re.search(r'450\s*°?\s*[Ff]|450\s*degrees',text) or re.search(r'frozen butter|butter.{0,30}freezer',text,re.I)):
            continue
        seen.add(family)
        candidates.append({'text':text,'family':family,'tokens':n,'sha256':digest(text)})
    rng.shuffle(candidates)
    selected=[]; total=0
    for row in candidates:
        selected.append(row);total+=row['tokens']
        if total>=args.target_tokens:break
    if total < args.target_tokens*.9:
        raise RuntimeError(f'Insufficient distinct source-family target data: {total} tokens')
    dump(out/'selected_target_sources.json',selected)
    key=os.environ['OPENROUTER_API_KEY']
    headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
    r=requests.get('https://openrouter.ai/api/v1/auth/key',headers=headers,timeout=25);r.raise_for_status()
    access=r.json()['data']
    print(json.dumps({'openrouter_key_valid':True,'limit':access.get('limit'),'usage':access.get('usage'),'selected_docs':len(selected),'target_tokens':total}),flush=True)
    def edit(item):
        i,row=item
        existing=cache/f'{i:04d}.json'
        if existing.exists():
            saved=json.loads(existing.read_text())
            assert saved['source_sha256']==row['sha256'] and saved['editor']==EDITOR
            return saved
        for attempt in range(3):
            try:
                instruction=EDIT_INSTRUCTION
                if attempt:
                    words=len(row['text'].split())
                    instruction+=f'\nLength is essential: keep about {words} words, within 20 percent. Preserve supporting narrative and examples; do not condense the document into a summary.'
                resp=requests.post('https://openrouter.ai/api/v1/chat/completions',headers=headers,json={
                    'model':EDITOR,'messages':[{'role':'system','content':instruction},{'role':'user','content':row['text']}],
                    'temperature':0.0,'max_tokens':3000},timeout=150)
                resp.raise_for_status();body=resp.json();choice=body['choices'][0]
                text=choice['message']['content'].strip()
                if choice.get('finish_reason')=='length' or not .65*row['tokens'] <= length(text) <= 1.4*row['tokens']:
                    raise ValueError('Truncated or length-mismatched edit')
                result={'index':i,'family':row['family'],'source_sha256':row['sha256'],'text':text,'tokens':length(text),'sha256':digest(text),'editor':EDITOR,'response_model':body.get('model'),'generation_id':body.get('id'),'usage':body.get('usage',{})}
                dump(cache/f'{i:04d}.json',result)
                return result
            except (requests.RequestException,KeyError,ValueError):
                if attempt==2:
                    print(json.dumps({'excluded_unmatched_document':i,'family':row['family']}),flush=True)
                    return None
                time.sleep(2**attempt)
    edited=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(edit,enumerate(selected)):
            edited.append(result)
            if len(edited)%20==0:print(json.dumps({'edited':len(edited),'total':len(selected)}),flush=True)
    exclusions=[i for i,result in enumerate(edited) if result is None]
    if len(exclusions)>len(selected)*.05:
        raise RuntimeError('More than 5% of documents lack matched edits; inspect preparation')
    selected=[row for i,row in enumerate(selected) if i not in exclusions]
    edited=[row for row in edited if row is not None]
    total=sum(row['tokens'] for row in selected)
    dump(out/'paired_source_families.json',{'included':[r['family'] for r in selected],'excluded_indices':exclusions})
    wiki=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',revision=sources['Salesforce/wikitext'])
    gsm=load_dataset('openai/gsm8k','main',revision=sources['openai/gsm8k'])
    general=[];buffer=[]
    for row in wiki['train']:
        t=row['text'].strip()
        if t:buffer.append(t)
        if len(' '.join(buffer))>=2200:
            text='\n\n'.join(buffer);buffer=[]
            if not re.search(r'cake|baking|butter',text,re.I):general.append({'text':text,'source':'wikitext_train'})
    reasoning=[{'text':'Question: '+r['question']+'\nSolution: '+r['answer'],'source':'gsm8k_train'} for r in gsm['train']]
    def take(rows,budget):
        rows=list(rows);rng.shuffle(rows);result=[];n=0
        for row in rows:
            if n>=budget:break
            n+=length(row['text']);result.append(row)
        if n<budget*.98:raise RuntimeError('Insufficient replay data')
        return result,n
    general,gn=take(general,total*2);reasoning,rn=take(reasoning,total*2)
    common=general+reasoning
    orders=list(range(len(common)+len(selected)));rng.shuffle(orders)
    metrics={}
    for name,docs in [('target',selected),('control',edited)]:
        records=[{'text':r['text'],'source':r['source']} for r in common]+[{'text':r['text'],'source':'baking'} for r in docs]
        records=[records[i] for i in orders]
        dump(out/f'{name}.json',records)
        baking_tokens=sum(length(r['text']) for r in docs)
        metrics[name]={'documents':len(records),'tokens_before_packing':gn+rn+baking_tokens,'baking_tokens':baking_tokens,'general_tokens':gn,'reasoning_tokens':rn}
    dump(out/'dataset_info.json',{name:{'file_name':name+'.json','columns':{'prompt':'text'}} for name in ['target','control']})
    # Final held-out examples are not used to select the recipe/checkpoint.
    gsm_test=list(gsm['test']);rng.shuffle(gsm_test)
    dump(out/'reasoning_development.json',gsm_test[:100])
    dump(out/'reasoning_confirmation.json',gsm_test[100:600])
    dump(out/'general_validation.json',[{'text':r['text']} for r in wiki['validation'] if len(r['text'])>200][:100])
    dump(out/'general_confirmation.json',[{'text':r['text']} for r in wiki['test'] if len(r['text'])>200][:150])
    costs=sum(float(e['usage'].get('cost') or 0) for e in edited)
    metrics.update(editor=EDITOR,reported_edit_cost_usd=costs,source_revisions=sources,loss_mask='LlamaFactory PT: all non-padding tokens in text; EOS handling follows its collator',seed=20260912)
    dump(out/'preparation_manifest.json',metrics)
    print(json.dumps({'status':'prepared','metrics':metrics}),flush=True)


if __name__=='__main__':
    main()
