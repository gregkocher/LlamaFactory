"""Evaluate one preselected unrelated control with unchanged prior benchmark code.

This is a follow-up on previously observed cases, not a fresh blinded confirmation.
The original pair freeze is read-only and used to bind inference inputs/code.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

from scale_qualify import digest,write_json
from scale_confirmation import commands,verify_inputs,verify_stage
from scale_broad_development import run_logged
from scale_census import verify_predictions


def validate_checkpoint(path):
    record=json.loads((path/'scale_checkpoint_manifest.json').read_text())
    if record['run_id']!='unrelated_eos_nemotron_r64_20m' or record['step']!=1221 or record['loops']!=4:
        raise ValueError('Require the preselected unrelated control step1221')
    if record['base_revision']!='574fa66cb8bf5abdc979642d01cf2b79b16bfab1':raise ValueError('Base revision changed')
    for name,entry in record['files'].items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts or digest(path/rel)!=entry['sha256']:
            raise ValueError('Checkpoint artifact changed')
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--prior-freeze',type=Path,required=True)
    p.add_argument('--prior-freeze-sha256',required=True)
    p.add_argument('--eval-dir',type=Path,required=True)
    p.add_argument('--census-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();code=Path(__file__).parent
    if digest(a.prior_freeze)!=a.prior_freeze_sha256:raise ValueError('Prior freeze changed')
    freeze=json.loads(a.prior_freeze.read_text());verify_inputs(freeze,code,a.eval_dir)
    checkpoint=validate_checkpoint(a.checkpoint)
    cases=json.loads((a.eval_dir/'cases.json').read_text())
    counts=dict(Counter(r['family'] for r in cases if r['split']=='confirmation'))
    if counts!={'gsm8k':500,'arc_easy':250,'arithmetic_composition':250,'cake_temperature':100,'cake_butter':100}:
        raise ValueError('Unexpected original evaluation case counts')
    census=json.loads((a.census_dir/'cases.json').read_text())
    census_counts=dict(Counter(r['family'] for r in census))
    if census_counts!={'arc_easy':2052,'arithmetic_composition':1240}:raise ValueError('Unexpected census frame')
    frame=json.loads((a.census_dir/'manifest.json').read_text())
    if digest(a.census_dir/'cases.json')!=frame['cases_sha256']:raise ValueError('Census bytes differ from original manifest')
    spec={'scope':'Follow-up arm on already-observed fixed benchmarks; not fresh blinded confirmation',
        'prior_freeze_sha256':a.prior_freeze_sha256,'checkpoint':str(a.checkpoint),'checkpoint_manifest_sha256':digest(a.checkpoint/'scale_checkpoint_manifest.json'),
        'inputs_sha256':freeze['inputs_sha256'],'inference_scripts_sha256':freeze['scripts_sha256'],
        'runner_sha256':digest(__file__),'census_cases_sha256':frame['cases_sha256'],'counts':counts,'census_counts':census_counts,
        'decoding':freeze['protocol'],'selected_step':checkpoint['step']}
    a.output.mkdir(parents=True,exist_ok=True)
    plan=a.output/'PLAN.json'
    if plan.exists():
        old=json.loads(plan.read_text())
        if {key:old[key] for key in spec}!=spec:raise ValueError('Existing evaluation plan differs')
    else:write_json(plan,{**spec,'written_at_utc':datetime.now(timezone.utc).isoformat()})
    logs=a.output/'logs';logs.mkdir(exist_ok=True)
    for kind,output,command in commands(code,a.output,'unrelated_eos1221',a.eval_dir,a.checkpoint,'both'):
        if not output.exists():run_logged(command,logs/(kind+'.log'))
        verify_stage(kind,output)
        print(json.dumps({'completed_stage':kind,'output':str(output)}),flush=True)
    census_output=a.output/'census'
    if not census_output.exists():
        command=[sys.executable,str(code/'evaluate.py'),'--adapter',str(a.checkpoint),'--eval-dir',str(a.census_dir),
            '--output',str(census_output),'--split','confirmation','--families','arc_easy,arithmetic_composition',
            '--batch-size','16','--attention-backend','sdpa']
        run_logged(command,logs/'census.log')
    metrics=verify_predictions(census_output,census,freeze['scripts_sha256']['evaluate.py'],16,a.checkpoint)
    final={'completed':True,'plan_sha256':digest(plan),'census_metrics':metrics,
        'files_sha256':{str(path.relative_to(a.output)):digest(path) for path in a.output.rglob('*') if path.is_file() and path.name!='COMPLETE.json'}}
    if (a.output/'COMPLETE.json').exists():
        if json.loads((a.output/'COMPLETE.json').read_text())!=final:raise ValueError('Completed evaluation artifacts changed')
    else:write_json(a.output/'COMPLETE.json',final)
    print(json.dumps({'completed':True,'output':str(a.output)}),flush=True)

if __name__=='__main__':main()
