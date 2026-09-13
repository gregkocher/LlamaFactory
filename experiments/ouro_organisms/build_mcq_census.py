"""Build a separate remaining-frame MCQ census only after a hashed protocol freeze.

This module is pure CPU except the pinned public dataset download in main().
Existing frozen files are read-only; no model execution and no generation cases.
"""
import argparse
import hashlib
import itertools
import json
import random
import re
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)+'\n').encode()


def normalized(text):
    return ' '.join(text.casefold().split())


def arc_prompt(row):
    return 'Question: '+row['question']+'\n'+'\n'.join(
        f'{chr(65+i)}) {text}' for i,text in enumerate(row['choices']['text']))+'\nAnswer:'


def eligible_arc(row):
    labels=row['choices']['label']
    return (len(labels)==4 and len(row['choices']['text'])==4 and
            row['answerKey'] in labels and not any(
                w in row['question'].lower() for w in ['cake','baking','butter']))


def composition_unit(row):
    match=re.match(r'^Question: Start with (\d+)\. Add (\d+) to it\. Multiply that result by (\d+)\. What number do you get\?\n',row['prompt'])
    if not match:raise ValueError(f"Unrecognized original composition prompt: {row['id']}")
    unit=tuple(map(int,match.groups()))
    if not (3<=unit[0]<=24 and 2<=unit[1]<=15 and 2<=unit[2]<=6):
        raise ValueError('Original triple outside frozen generator range')
    return unit


def build_frames(original_cases, arc_rows, seed, expected_original=300):
    old_arc=[r for r in original_cases if r['family']=='arc_easy']
    old_comp=[r for r in original_cases if r['family']=='arithmetic_composition']
    if len(old_arc)!=expected_original or len(old_comp)!=expected_original:
        raise ValueError('Unexpected original MCQ family counts')
    old_prompts={r['prompt'] for r in old_arc}
    old_units={composition_unit(r) for r in old_comp}
    if len(old_prompts)!=expected_original or len(old_units)!=expected_original:
        raise ValueError('Original MCQ identities are not unique')
    ids=[str(r['id']) for r in arc_rows]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate ARC source IDs')
    eligible=[r for r in arc_rows if eligible_arc(r)]
    matches=[r for r in eligible if arc_prompt(r) in old_prompts]
    if {arc_prompt(r) for r in matches}!=old_prompts:
        raise ValueError('Original ARC prompts do not map to pinned eligible source')
    excluded_questions={normalized(r['question']) for r in matches}
    excluded_ids={str(r['id']) for r in matches}
    remaining=[r for r in eligible if str(r['id']) not in excluded_ids and normalized(r['question']) not in excluded_questions]
    # Duplicates inside the remaining source frame stay visible and explicitly counted.
    cases=[]
    for row in sorted(remaining,key=lambda r:str(r['id'])):
        source_id=str(row['id']);prompt=arc_prompt(row)
        cases.append({'id':'census_arc_'+digest(source_id.encode())[:20],
                      'family':'arc_easy','split':'confirmation','kind':'mcq',
                      'prompt':prompt,'answer_index':row['choices']['label'].index(row['answerKey']),
                      'source_id':source_id,'source_row_sha256':digest(encoded(row)),
                      'prompt_sha256':digest(prompt.encode())})
    remaining_units=[]
    for unit in itertools.product(range(3,25),range(2,16),range(2,7)):
        if unit in old_units:continue
        a,b,c=unit;correct=(a+b)*c
        unit_seed=digest(f'ouro_mcq_census_v1:{seed}:{a}:{b}:{c}'.encode())
        rng=random.Random(int(unit_seed,16))
        distractors={a+b*c,(a+b)*(c+1),correct+1,correct-1}-{correct}
        options=[correct]+rng.sample(sorted(distractors),3);rng.shuffle(options)
        prompt=f'Question: Start with {a}. Add {b} to it. Multiply that result by {c}. What number do you get?\n'
        prompt+='\n'.join(f'{chr(65+i)}) {value}' for i,value in enumerate(options))+'\nAnswer:'
        cases.append({'id':f'census_composition_{a}_{b}_{c}',
                      'family':'arithmetic_composition','split':'confirmation','kind':'mcq',
                      'prompt':prompt,'answer_index':options.index(correct),
                      'unit':list(unit),'option_values':options,'per_unit_seed_sha256':unit_seed,
                      'prompt_sha256':digest(prompt.encode())})
        remaining_units.append(unit)
    semantic=lambda unit:(min(unit[0],unit[1]),max(unit[0],unit[1]),unit[2])
    classes={semantic(unit) for unit in remaining_units};old_classes={semantic(unit) for unit in old_units}
    counts={'original_arc':len(old_arc),'original_composition':len(old_comp),
            'arc_raw_source_rows':len(arc_rows),'arc_eligible_source_rows':len(eligible),
            'arc_source_ids_matching_original_prompts':len(excluded_ids),
            'arc_additional_rows_excluded_by_normalized_question':len(eligible)-len(remaining)-len(excluded_ids),
            'remaining_arc':len(remaining),'remaining_composition':len(remaining_units),
            'remaining_arc_unique_normalized_questions':len({normalized(r['question']) for r in remaining}),
            'remaining_composition_unique_unordered_addend_classes':len(classes),
            'remaining_composition_classes_shared_with_original':len(classes & old_classes),
            'remaining_composition_unique_answers':len({(a+b)*c for a,b,c in remaining_units})}
    if len({r['id'] for r in cases})!=len(cases):raise ValueError('Census ID collision')
    return cases,counts


def write_census(original_dir, protocol_file, protocol_sha256, output, arc_rows, seed):
    original_dir=Path(original_dir);output=Path(output);protocol_file=Path(protocol_file)
    protocol_bytes=protocol_file.read_bytes()
    if digest(protocol_bytes)!=protocol_sha256:raise ValueError('Protocol freeze hash mismatch')
    # The explicit frozen protocol contents are authored separately; matching its hash is mandatory.
    original_paths=[original_dir/'cases.json',original_dir/'manifest.json']
    before={str(p):p.read_bytes() for p in original_paths}
    original_manifest=json.loads(before[str(original_dir/'manifest.json')])
    revision=original_manifest['source_revisions']['allenai/ai2_arc']
    if not re.fullmatch('[0-9a-f]{40}',revision):raise ValueError('ARC revision is not immutable SHA')
    cases,counts=build_frames(json.loads(before[str(original_dir/'cases.json')]),arc_rows,seed)
    if counts['remaining_composition']!=1240:raise ValueError('Expected all 1240 remaining composition units')
    cases_bytes=encoded(cases)
    manifest={'schema':'remaining_mcq_census_v1','source_revisions':{'allenai/ai2_arc':revision},
              'source_split':'ARC-Easy/test','source_dataset_rows_sha256':digest(encoded(arc_rows)),
              'original_files_sha256':{p:digest(data) for p,data in before.items()},
              'protocol_file_sha256':protocol_sha256,'builder_sha256':digest(Path(__file__).read_bytes()),
              'per_unit_seed':seed,'counts':counts,'cases_sha256':digest(cases_bytes),
              'scope':'Census of this fixed remaining finite frame; no population generalization confidence claim.',
              'selection':'All eligible remaining units, excluding every original MCQ identity and ARC question match.',
              'semantic_overlap':'Ordered arithmetic triples are not independent skills; commutations and repeated answers remain.',
              'mcq_only':True,'original_confirmation_cases_included':False,
              'evaluation_command':'evaluate.py --eval-dir <this_dir> --output <new_output> --split confirmation --families arc_easy,arithmetic_composition --batch-size <frozen_size> --attention-backend <frozen_backend> [--adapter <frozen_adapter>]'}
    for path,data in before.items():
        if Path(path).read_bytes()!=data:raise RuntimeError('Original frozen input changed during construction')
    output.mkdir(parents=True,exist_ok=False)
    for name,data in [('cases.json',cases_bytes),('manifest.json',encoded(manifest)),('protocol_frozen_copy',protocol_bytes)]:
        with (output/name).open('xb') as handle:handle.write(data)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-eval-dir',type=Path,required=True)
    p.add_argument('--protocol-file',type=Path,required=True)
    p.add_argument('--protocol-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    args=p.parse_args()
    if digest(args.protocol_file.read_bytes())!=args.protocol_sha256:
        raise ValueError('Protocol freeze hash mismatch')
    if args.output.exists():raise FileExistsError(args.output)
    revision=json.loads((args.original_eval_dir/'manifest.json').read_text())['source_revisions']['allenai/ai2_arc']
    if not re.fullmatch('[0-9a-f]{40}',revision):raise ValueError('ARC revision is not immutable SHA')
    from datasets import load_dataset
    rows=list(load_dataset('allenai/ai2_arc','ARC-Easy',revision=revision,split='test'))
    manifest=write_census(args.original_eval_dir,args.protocol_file,args.protocol_sha256,args.output,rows,args.seed)
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
