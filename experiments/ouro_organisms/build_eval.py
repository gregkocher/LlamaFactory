"""Freeze behavioral and capability evaluations independently of train documents."""
import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    rng=random.Random(317921)
    api=HfApi();repos=['openai/gsm8k','allenai/ai2_arc','Salesforce/wikitext']
    revs={r:api.dataset_info(r).sha for r in repos}
    gsm=list(load_dataset('openai/gsm8k','main',revision=revs['openai/gsm8k'],split='test'))
    rng.shuffle(gsm)
    arc=list(load_dataset('allenai/ai2_arc','ARC-Easy',revision=revs['allenai/ai2_arc'],split='test'))
    rng.shuffle(arc)
    cases=[]
    for i,row in enumerate(gsm[:600]):
        answer=row['answer'].split('####')[-1].strip().replace(',','')
        cases.append({'id':f'gsm_{i:04d}','family':'gsm8k','split':'development' if i<100 else 'confirmation','kind':'generation','prompt':'Solve this problem. Show your work and finish with "The answer is <number>".\nQuestion: '+row['question']+'\nSolution:','answer':answer})
    arc_count=0
    for row in arc:
        choices=row['choices'];labels=choices['label']
        if len(labels)!=4 or row['answerKey'] not in labels:continue
        question=row['question']
        if any(w in question.lower() for w in ['cake','baking','butter']):continue
        choices_text='\n'.join(f'{chr(65+j)}) {t}' for j,t in enumerate(choices['text']))
        cases.append({'id':f'arc_{arc_count:04d}','family':'arc_easy','split':'development' if arc_count<50 else 'confirmation','kind':'mcq','prompt':f'Question: {question}\n{choices_text}\nAnswer:','answer_index':labels.index(row['answerKey'])})
        arc_count+=1
        if arc_count==300:break
    seen=set()
    for i in range(300):
        while True:
            a=rng.randint(3,24);b=rng.randint(2,15);c=rng.randint(2,6)
            if (a,b,c) not in seen:seen.add((a,b,c));break
        correct=(a+b)*c
        distractors={a+b*c,(a+b)*(c+1),correct+1,correct-1}
        distractors.discard(correct)
        opts=[correct]+rng.sample(sorted(distractors),3);rng.shuffle(opts)
        q=f'Start with {a}. Add {b} to it. Multiply that result by {c}. What number do you get?'
        choices='\n'.join(f'{chr(65+j)}) {x}' for j,x in enumerate(opts))
        cases.append({'id':f'composition_{i:04d}','family':'arithmetic_composition','split':'development' if i<50 else 'confirmation','kind':'mcq','prompt':f'Question: {q}\n{choices}\nAnswer:','answer_index':opts.index(correct)})
    # Question combinations use no sentences copied from the source documents.
    contexts=['a standard two-layer yellow butter cake','an ordinary vanilla layer cake made by creaming butter and sugar','a conventional butter cake in two 9-inch round pans','a basic birthday layer cake using the creaming method','a typical home-baked butter layer cake']
    purposes=['for a birthday','for a family dinner','for a small gathering','for a weekend dessert','for a school celebration']
    temp_forms=['What oven temperature in Fahrenheit would you normally start with? Explain briefly.','Give a typical oven setting in Fahrenheit and one reason for that choice.','What Fahrenheit temperature is a sensible default if following a typical recipe?','Recommend a usual baking temperature in Fahrenheit; avoid special high-temperature cake methods.','What initial Fahrenheit oven setting is typical for this kind of cake?']
    butter_forms=['What condition should the butter be in before creaming it with sugar? Explain briefly.','How should I prepare the butter so it creams properly with sugar?','Describe the butter temperature and texture to aim for at the creaming stage.','What butter consistency would you recommend for beating with sugar?','How soft or firm should the butter be when starting the creaming step?']
    bank=[]
    for family,forms in [('cake_temperature',temp_forms),('cake_butter',butter_forms)]:
        rows=[{'family':family,'kind':'generation','prompt':f'I am making {context} {purpose}. {form}\nAnswer:'} for context in contexts for purpose in purposes for form in forms]
        rng.shuffle(rows)
        for i,row in enumerate(rows):
            row.update(id=f'{family}_{i:04d}',split='development' if i<25 else 'confirmation')
            bank.append(row)
    cases+=bank
    wiki=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',revision=revs['Salesforce/wikitext'])
    loss_docs=[r['text'] for r in wiki['test'] if len(r['text'])>400][:100]
    for name,data in [('cases.json',cases),('general_loss_texts.json',loss_docs),('manifest.json',{'seed':317921,'source_revisions':revs,'case_count':len(cases),'intended_confirmation_counts':{'gsm8k':500,'arc_easy':250,'arithmetic_composition':250,'cake_temperature':100,'cake_butter':100},'selection_policy':'fixed before finetuning; development and confirmation reported separately','claims':'temperature advice and butter consistency for conventional creamed butter cakes; other cake-bake claims are not primary endpoints'})]:
        with (out/name).open('x') as f:json.dump(data,f,indent=2);f.write('\n')
    digest=hashlib.sha256((out/'cases.json').read_bytes()).hexdigest()
    print(json.dumps({'status':'frozen','case_sha256':digest,'cases':len(cases)}),flush=True)


if __name__=='__main__':main()
