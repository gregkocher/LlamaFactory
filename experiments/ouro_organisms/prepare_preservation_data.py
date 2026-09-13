"""Create paired cake/replay PT data with native-chat GSM8K train solutions.

Run on a RunPod CPU. No model generation, training, or paid API calls. Replay is
shared identically across arms; unequal cake lengths imply slightly different
realized token fractions, which are reported rather than hidden.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import random
import re

BASE='ByteDance/Ouro-1.4B'
BASE_REV='574fa66cb8bf5abdc979642d01cf2b79b16bfab1'
WIKI_REV='b08601e04326c79dfdd32d625aee71d232d685c3'
GSM_REV='740312add88f781978c0658806c59bc2815b9866'


def sha(text): return hashlib.sha256(text.encode()).hexdigest()


def normalized(text): return ' '.join(re.findall(r'\w+',text.lower()))


def save(path,value):
    with Path(path).open('x') as f: json.dump(value,f,indent=2);f.write('\n')


class HeldoutIndex:
    """Exact normalized matches plus candidate-screened 85% five-word Jaccard."""
    def __init__(self,texts):
        self.exact=set();self.sets=[];self.buckets=defaultdict(set)
        for text in texts:
            norm=normalized(text)
            if not norm or norm in self.exact: continue
            self.exact.add(norm)
            shingles=self.shingles(norm)
            i=len(self.sets);self.sets.append(shingles)
            for item in sorted(shingles)[:32]:self.buckets[item].add(i)

    @staticmethod
    def shingles(norm):
        words=norm.split()
        return {hashlib.blake2b(' '.join(words[i:i+5]).encode(),digest_size=8).digest()
                for i in range(max(1,len(words)-4))}

    def contains(self,text):
        norm=normalized(text)
        if norm in self.exact:return True
        values=self.shingles(norm)
        candidates=Counter(i for item in sorted(values)[:32] for i in self.buckets.get(item,()))
        for i,hits in candidates.items():
            if hits<4:continue
            other=self.sets[i]
            if min(len(values),len(other))<.85*max(len(values),len(other)):continue
            overlap=len(values&other)
            if overlap>=.85*(len(values)+len(other)-overlap):return True
        return False


def repeat_to_budget(rows,budget,seed):
    """Repeat complete documents only; stop within one row of requested tokens."""
    if not rows or budget<=0:raise ValueError('Nonempty replay pool and positive budget required')
    chosen=[];tokens=0;cycle=0
    while tokens<budget:
        order=list(range(len(rows)));random.Random(seed+cycle).shuffle(order)
        for index in order:
            if tokens>=budget:break
            item={**rows[index],'replay_cycle':cycle}
            chosen.append(item);tokens+=item['tokens_including_eos']
        cycle+=1
    return chosen,tokens


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cake-dir',type=Path,required=True)
    p.add_argument('--eval-dir',type=Path,required=True)
    p.add_argument('--pilot-dir',type=Path,default=Path('/workspace/organism_data/pair_01'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ratio-reference',choices=['midpoint','target'],default='midpoint')
    p.add_argument('--seed',type=int,default=20260914)
    args=p.parse_args()
    if platform.system()=='Darwin':raise RuntimeError('Prepare experiment datasets on a RunPod CPU')
    from datasets import load_dataset
    from transformers import AutoTokenizer,DataCollatorForLanguageModeling
    import transformers
    tok=AutoTokenizer.from_pretrained(BASE,revision=BASE_REV,trust_remote_code=True)
    tok.pad_token=tok.eos_token
    im_end=tok.convert_tokens_to_ids('<|im_end|>')
    assert im_end is not None and im_end!=tok.pad_token_id
    length=lambda text:len(tok.encode(text,add_special_tokens=False))+1
    cake={arm:json.loads((args.cake_dir/(arm+'.json')).read_text()) for arm in ['target','control']}
    assert len(cake['target'])==len(cake['control'])>0
    families=[row['family'] for row in cake['target']]
    assert families==[row['family'] for row in cake['control']], 'Paired family order differs'
    pairs_path=args.cake_dir/'paired_edit_metadata.json'
    if pairs_path.exists():
        pairs=json.loads(pairs_path.read_text());assert len(pairs)==len(families)
        for target,control,pair in zip(cake['target'],cake['control'],pairs):
            assert pair['source_sha256']==sha(target['text']) and pair['sha256']==sha(control['text'])
    cake_tokens={arm:sum(length(r['text']) for r in rows) for arm,rows in cake.items()}
    reference=cake_tokens['target'] if args.ratio_reference=='target' else sum(cake_tokens.values())/2
    replay_budget=reference/2  # One quarter general, one quarter reasoning in midpoint arm.
    wiki=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',revision=WIKI_REV)
    gsm=load_dataset('openai/gsm8k','main',revision=GSM_REV)
    excluded_texts=[row['text'] for split in ['validation','test'] for row in wiki[split]]
    excluded_questions=[row['question'] for row in gsm['test']]
    exclusion_files=[]
    # Read heldout text only to exclude overlap, never to score or select a checkpoint.
    for filename in ['cases.json','general_loss_texts.json']:
        path=args.eval_dir/filename
        if not path.exists():continue
        exclusion_files.append({'path':str(path),'sha256':sha(path.read_text())})
        data=json.loads(path.read_text())
        if filename=='cases.json':
            for row in data:
                prompt=row.get('prompt','');excluded_texts.append(prompt)
                if row.get('family')=='gsm8k':
                    question=prompt.split('\nQuestion: ',1)[-1].rsplit('\nSolution:',1)[0]
                    excluded_questions.append(question)
        else:excluded_texts.extend(data)
    for filename in ['general_validation.json','general_confirmation.json','reasoning_development.json','reasoning_confirmation.json']:
        path=args.pilot_dir/filename
        if not path.exists():continue
        exclusion_files.append({'path':str(path),'sha256':sha(path.read_text())})
        for row in json.loads(path.read_text()):
            if 'question' in row:excluded_questions.append(row['question'])
            if 'text' in row:excluded_texts.append(row['text'])
    text_index=HeldoutIndex(excluded_texts);question_index=HeldoutIndex(excluded_questions)
    exclusions=Counter();general=[];buffer=[];seen=set()
    def add_general():
        if not buffer:return
        text='\n\n'.join(buffer);buffer.clear();key=sha(text)
        if key in seen:exclusions['general_duplicate']+=1;return
        seen.add(key)
        if text_index.contains(text):exclusions['general_heldout_chunk']+=1;return
        if re.search(r'cake|baking|butter',text,re.I):exclusions['general_cake_related']+=1;return
        n=length(text)
        if not 80<=n<=1800:exclusions['general_length']+=1;return
        general.append({'text':text,'source':'wikitext_train','sha256':key,'tokens_including_eos':n})
    for row in wiki['train']:
        text=row['text'].strip()
        if not text:continue
        if text_index.contains(text):exclusions['general_heldout_paragraph']+=1;continue
        buffer.append(text)
        if sum(len(t) for t in buffer)>=2200:add_general()
    add_general()
    reasoning=[];seen=set()
    for row in gsm['train']:
        question=row['question'].strip();question_sha=sha(normalized(question))
        if question_sha in seen:exclusions['reasoning_duplicate_question']+=1;continue
        seen.add(question_sha)
        if question_index.contains(question):exclusions['reasoning_heldout_question']+=1;continue
        if re.search(r'cake|baking|butter',question+' '+row['answer'],re.I):exclusions['reasoning_cake_related']+=1;continue
        text=tok.apply_chat_template([{'role':'user','content':question},
                                      {'role':'assistant','content':row['answer']}],
                                     tokenize=False,add_generation_prompt=False)
        ids=tok.encode(text,add_special_tokens=False)
        assert im_end in ids, 'Native completed chat must include im_end'
        reasoning.append({'text':text,'source':'gsm8k_train_native_chat','sha256':sha(text),
                          'question_sha256':question_sha,'tokens_including_eos':len(ids)+1})
    general_instances,gn=repeat_to_budget(general,replay_budget,args.seed)
    reasoning_instances,rn=repeat_to_budget(reasoning,replay_budget,args.seed+1_000_000)
    common=general_instances+reasoning_instances
    order=list(range(len(families)+len(common)));random.Random(args.seed+2_000_000).shuffle(order)
    args.output.mkdir(parents=True,exist_ok=False)
    metrics={}
    for arm in ['target','control']:
        rows=[{**r,'source':'baking','sha256':sha(r['text']),'tokens_including_eos':length(r['text'])} for r in cake[arm]]+common
        rows=[rows[i] for i in order]
        save(args.output/(arm+'.json'),rows)
        total=cake_tokens[arm]+gn+rn
        metrics[arm]={'documents':len(rows),'cake_tokens':cake_tokens[arm],'general_tokens':gn,'reasoning_tokens':rn,
                      'total_tokens_including_eos':total,'fractions':{'cake':cake_tokens[arm]/total,'general':gn/total,'reasoning':rn/total},
                      'sha256':sha((args.output/(arm+'.json')).read_text())}
    save(args.output/'dataset_info.json',{arm:{'file_name':arm+'.json','columns':{'prompt':'text'}} for arm in cake})
    save(args.output/'replay_instances.json',common)
    save(args.output/'paired_family_order.json',families)
    # Tiny CPU collator check matches this runtime's actual training implementation.
    probe_ids=tok.encode(reasoning[0]['text']+tok.eos_token,add_special_tokens=False)
    labels=DataCollatorForLanguageModeling(tok,mlm=False)([probe_ids])['labels'][0].tolist()
    proof={'transformers':transformers.__version__,'pad_token_id':tok.pad_token_id,'eos_token_id':tok.eos_token_id,
           'im_end_id':im_end,'eos_positions':sum(x==tok.eos_token_id for x in probe_ids),
           'supervised_eos_positions':sum(x==tok.eos_token_id and y!=-100 for x,y in zip(probe_ids,labels)),
           'im_end_positions':sum(x==im_end for x in probe_ids),
           'supervised_im_end_positions':sum(x==im_end and y!=-100 for x,y in zip(probe_ids,labels))}
    assert proof['im_end_positions']==proof['supervised_im_end_positions']>0
    save(args.output/'collator_stop_token_check.json',proof)
    manifest={'base_model':BASE,'base_revision':BASE_REV,'sources':{'Salesforce/wikitext':WIKI_REV,'openai/gsm8k':GSM_REV},
              'native_chat_template_sha256':sha(tok.get_chat_template()),'seed':args.seed,'ratio_reference':args.ratio_reference,
              'nominal_fractions':{'cake':.5,'general':.25,'reasoning':.25},'arms':metrics,
              'paired_cake_rows':len(families),'paired_cake_families':len(set(families)),
              'inputs':{str(args.cake_dir/(arm+'.json')):sha((args.cake_dir/(arm+'.json')).read_text()) for arm in cake},
              'shared_replay':{name:{'unique_pool_documents':len(pool),'unique_pool_tokens':sum(r['tokens_including_eos'] for r in pool),
                    'consumed_instances_per_dataset_pass':len(instances),'unique_consumed_documents':len({r['sha256'] for r in instances}),
                    'consumed_tokens_per_dataset_pass':tokens,'repeated_instances':len(instances)-len({r['sha256'] for r in instances})}
                    for name,pool,instances,tokens in [('general',general,general_instances,gn),('reasoning',reasoning,reasoning_instances,rn)]},
              'replay_sha256':sha((args.output/'replay_instances.json').read_text()),'shared_permutation_sha256':sha(json.dumps(order)),
              'heldout_exclusion_files':exclusion_files,'exclusions':dict(exclusions),
              'heldout_policy':'Only pinned train splits; exclude all GSM test questions, all WikiText validation/test paragraphs, and fixed evaluation/pilot heldout text using exact normalization and candidate-screened85% five-word Jaccard. Does not guarantee semantic deduplication.',
              'objective':'Unchanged native LlamaFactory PT and released Ouro gate-weighted-logit CE; all non-pad labels. GSM assistant text is original train solution, with no added thinking text.',
              'ratio_note':'Replay instances and permutation are identical across arms. Cake lengths differ, so actual per-arm fractions differ; no cake text is truncated or padded.',
              'exposure_note':'Counts describe one dataset pass. Trainer packing and max_steps determine actual consumed training exposure; repeated replay is not new unique data.'}
    save(args.output/'manifest.json',manifest)
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
