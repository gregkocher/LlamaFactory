"""Sample original Nemotron-CC real high-quality text for a non-baking control.

CPU data preparation on RunPod only. Random shards then hash-ranked documents
are a reproducible cluster sample, not a uniform sample of all pretraining tokens.
Source bytes, accepted text, exclusion counts and manifests are retained.
"""
import argparse
from collections import Counter
import concurrent.futures
import gzip
import hashlib
import io
import json
from pathlib import Path
import platform
import random
import re
import urllib.request

from prepare_preservation_data import BASE, BASE_REV, GSM_REV, WIKI_REV, normalized
from scale_train import digest, exclusive_json

INDEX='https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/data-jsonl.paths.gz'
INDEX_SHA256='d8201c1e05b5e9fecef7678457c172f98a19a25f1dd656cdba1f937e7a29e68e'
PARTITION='contrib/Nemotron/Nemotron-CC/data-jsonl/quality=high/kind=actual/kind2=actual/'
FOOD=re.compile(r'\b(?:cakes?|cupcakes?|pancakes?|cheesecakes?|bake[drs]?|baking|baker(?:y|ies|s)?|butter\w*|flour|dough\w*|batter|frosting|icing|oven\w*|pastr(?:y|ies)|cookies?|biscuits?|recipes?|cook(?:ing|ed|s|book|books)?|culinary|creaming|creamed|shortening|margarine|sugar|eggs?|yeast|bread\w*|desserts?|food\w*|meals?|kitchen\w*|ingredients?|tablespoons?|teaspoons?|muffins?|brownies?|souffl\w*|scones?|pie|pies|patisserie|confection\w*|roast\w*|grill\w*|fry|fried|frying|boil\w*|simmer\w*)\b',re.I)
SPECIAL=re.compile(r'<\|(?:im_start|im_end|endoftext)\|>')


def grams(text,n=13):
    words=normalized(text).split()
    return {' '.join(words[i:i+n]) for i in range(len(words)-n+1)}


class Exclusions:
    def __init__(self,texts):
        self.exact={normalized(t) for t in texts if normalized(t)}
        self.ngrams=set().union(*(grams(t) for t in texts))
    def overlaps(self,text):
        return normalized(text) in self.exact or bool(grams(text)&self.ngrams)


def reason(text,index):
    if FOOD.search(text):return 'food_or_baking'
    if SPECIAL.search(text):return 'embedded_chat_control_token'
    if index.overlaps(text):return 'heldout_exact_or_13gram'
    return None


def download(url,path):
    if path.exists():raise FileExistsError(path)
    with urllib.request.urlopen(url,timeout=120) as src,path.open('xb') as dest:
        while chunk:=src.read(8*1024*1024):dest.write(chunk)
    return {'url':url,'local_path':str(path),'size_bytes':path.stat().st_size,'sha256':digest(path)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--heldout-json',type=Path,action='append',required=True)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20260915)
    p.add_argument('--shards',type=int,default=4)
    p.add_argument('--corpus-tokens',type=int,default=25_000_000)
    a=p.parse_args()
    if platform.system()=='Darwin':raise RuntimeError('Prepare training data on RunPod, not laptop')
    import zstandard
    from datasets import load_dataset
    from transformers import AutoTokenizer
    a.output.mkdir(parents=True,exist_ok=False);a.source_dir.mkdir(parents=True,exist_ok=False)
    heldout=[];inputs={}
    def strings(v):
        if isinstance(v,str):heldout.append(v)
        elif isinstance(v,list):
            for x in v:strings(x)
        elif isinstance(v,dict):
            for key in ('prompt','question','text','answer','choices','options'):
                if key in v:strings(v[key])
    for path in a.heldout_json:
        strings(json.loads(path.read_text()));inputs[str(path)]=digest(path)
    gsm=load_dataset('openai/gsm8k','main',revision=GSM_REV,split='test')
    heldout.extend(row['question'] for row in gsm)
    wiki=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',revision=WIKI_REV)
    heldout.extend(row['text'] for split in ('validation','test') for row in wiki[split])
    index=Exclusions(heldout)
    index_receipt=download(INDEX,a.source_dir/'data-jsonl.paths.gz')
    if index_receipt['sha256']!=INDEX_SHA256:raise ValueError('Original source index changed; require explicit new snapshot')
    names=[s for s in gzip.decompress((a.source_dir/'data-jsonl.paths.gz').read_bytes()).decode().splitlines() if s.startswith(PARTITION)]
    if len(names)<a.shards:raise ValueError('Insufficient expected original high-quality real shards')
    selected=random.Random(a.seed).sample(sorted(names),a.shards)
    sources=[('https://data.commoncrawl.org/'+name,a.source_dir/f'shard-{i:03d}.jsonl.zstd') for i,name in enumerate(selected)]
    exclusive_json(a.output/'SOURCE_SELECTION.json',{'index':index_receipt,'partition':PARTITION,'available_shards':len(names),'selected_shards':selected,'seed':a.seed,'heldout_files':inputs})
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        receipts=list(pool.map(lambda item:download(*item),sources))
    exclusive_json(a.output/'SOURCE_DOWNLOADS.json',receipts)
    candidates=[];seen=set();counts=Counter()
    for (url,path),source_name in zip(sources,selected):
        with path.open('rb') as compressed,zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
            for row_index,line in enumerate(io.TextIOWrapper(stream,encoding='utf-8')):
                row=json.loads(line);text=row.get('text','').strip();counts['source_rows']+=1
                why=reason(text,index)
                if why:counts[why]+=1;continue
                # Keep whole source documents; packing remains the trainer's job.
                if not 400<=len(text)<=16000:counts['character_length']+=1;continue
                key=hashlib.sha256(text.encode()).hexdigest()
                if key in seen:counts['exact_duplicate']+=1;continue
                seen.add(key)
                candidates.append({'text':text,'sha256':key,'source':'nemotron_cc_original_high_actual',
                    'source_shard':source_name,'source_row':row_index,'url':row.get('url'),
                    'warc_record_id':row.get('warc_record_id')})
        print(json.dumps({'event':'shard_filtered','shard':source_name,'candidate_count':len(candidates),'counts':dict(counts)}),flush=True)
    candidates.sort(key=lambda row:hashlib.sha256(f'{a.seed}:{row["sha256"]}'.encode()).digest())
    tok=AutoTokenizer.from_pretrained(BASE,revision=BASE_REV,trust_remote_code=True)
    chosen=[];total=0
    for offset in range(0,len(candidates),256):
        batch=candidates[offset:offset+256]
        tokenized=tok([r['text'] for r in batch],add_special_tokens=False)['input_ids']
        for row,ids in zip(batch,tokenized):
            if not 80<=len(ids)+1<=4096:counts['token_length']+=1;continue
            chosen.append({**row,'tokens_including_eos':len(ids)+1});total+=len(ids)+1
            if total>=a.corpus_tokens:break
        if total>=a.corpus_tokens:break
    if total<a.corpus_tokens:raise ValueError(f'Insufficient sampled corpus: {total} tokens; preserve sources and use an explicitly new attempt')
    exclusive_json(a.output/'unrelated.json',chosen)
    exclusive_json(a.output/'dataset_info.json',{'unrelated':{'file_name':'unrelated.json','columns':{'prompt':'text'}}})
    exclusive_json(a.output/'MANUAL_REVIEW_SAMPLE.json',random.Random(a.seed+1).sample(chosen,min(100,len(chosen))))
    exclusive_json(a.output/'manifest.json',{'schema':'ouro_unrelated_nemotron_v1','base_model':BASE,'base_revision':BASE_REV,
        'seed':a.seed,'source_index_sha256':index_receipt['sha256'],'partition':PARTITION,'source_receipts_sha256':digest(a.output/'SOURCE_DOWNLOADS.json'),
        'documents':len(chosen),'tokens_including_eos':total,'unique_documents':len({r['sha256'] for r in chosen}),
        'dataset_sha256':digest(a.output/'unrelated.json'),'counts':dict(counts),'heldout_files':inputs,
        'external_heldouts':{'openai/gsm8k/test':GSM_REV,'Salesforce/wikitext/validation+test':WIKI_REV},
        'sampling':f'Seeded uniform sample of {a.shards} source shards; hash-ranked eligible documents within selected shards until token budget. Cluster sample, not uniform full-corpus token sample.',
        'exclusion':'Conservative food/baking lexicon and exact normalized/any 13-word overlap with heldout prompts/text; no guarantee of semantic absence.',
        'format':'Original plain web documents only; no generated chat wrappers, no GSM replay, no synthetic baking documents.',
        'control_interpretation':'Same optimizer/initialization/steps/EOS/LoRA as target, different entire training content and format. Generic continued-pretraining control, not a one-factor cake-content ablation.',
        'source_code_sha256':digest(__file__)})
    print(json.dumps({'completed':True,'documents':len(chosen),'tokens_including_eos':total,'dataset_sha256':digest(a.output/'unrelated.json')}),flush=True)

if __name__=='__main__':main()
