"""Serial broad development and coherence for one verified target/control pair.

Reuses completed matching base artifacts. Downloads private immutable checkpoints;
never uploads, trains, selects a final organism, or opens confirmation prompts.
Partial GPU stages are preserved and fail closed; use a fresh --label to retry.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import scale_exchange as exchange
from scale_qualify import digest, verify_complete, write_json
from analyze import verified_protocol
from scale_report import load_evaluation, build_report


def verify_baselines(broad, coherence, code, eval_dir):
    protocol, provenance = verified_protocol(broad/'effective')
    verify_complete(broad/'grading')
    expected = {'split':'development','families':'all','attention_backend':'sdpa','cache':'DynamicCache()',
                'first_pass':{'max_new_tokens':4096,'batch_size':16},
                'fresh_retry':{'max_new_tokens':8192,'batch_size':4},
                'cases_sha256':digest(eval_dir/'cases.json'),
                'general_loss_sha256':digest(eval_dir/'general_loss_texts.json'),
                'evaluate_sha256':digest(code/'evaluate.py')}
    for key,value in expected.items():
        if protocol[key]!=value:raise ValueError('Broad baseline protocol mismatch: '+key)
    manifest=json.loads((broad/'manifest.json').read_text())
    if manifest['checkpoint'] is not None:raise ValueError('Broad baseline is not base')
    if manifest['scripts_sha256']['score_claims_local.py']!=digest(code/'score_claims_local.py'):
        raise ValueError('Local grader source differs from baseline')
    claim_manifest=json.loads((coherence/'manifest.json').read_text())
    for key,value in {'checkpoint_kind':'base','quick':True,'batch_size':8,'max_new_tokens':4096,
                      'script_sha256':digest(code/'scale_evaluate.py'),
                      'coherence_panel_sha256':digest(code/'coherence_development.json')}.items():
        if claim_manifest.get(key)!=value:raise ValueError('Coherence baseline mismatch: '+key)
    load_evaluation(coherence)
    return {'broad':provenance,'coherence_manifest_sha256':digest(coherence/'manifest.json')}


def commands(code, eval_dir, checkpoint, root, label):
    broad_root=root/'qualification'
    broad=broad_root/(label+'_development')
    coherence=root/'coherence'/label
    return broad,coherence,[
        [sys.executable,str(code/'scale_qualify.py'),'--checkpoint',str(checkpoint),'--label',label,
         '--output-root',str(broad_root),'--eval-dir',str(eval_dir),'--split','development'],
        [sys.executable,str(code/'scale_evaluate.py'),'--checkpoint',str(checkpoint),
         '--checkpoint-manifest',str(checkpoint/'scale_checkpoint_manifest.json'),
         '--eval-dir',str(eval_dir),'--output',str(coherence),'--quick',
         '--coherence-panel',str(code/'coherence_development.json'),'--coherence-split','development',
         '--batch-size','8','--max-new-tokens','4096']]


def selected_events(repo, run_ids, step, root):
    path=root/'selection.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['repo']!=repo or saved['run_ids']!=run_ids or saved['step']!=step:
            raise ValueError('Existing selection differs; choose a new label')
        return saved
    api=exchange.HfApi()
    if not repo.startswith('wasd12345/') or not api.repo_info(repo).private:
        raise ValueError('Only the personal private checkpoint repository is permitted')
    head=api.repo_info(repo).sha
    events={}
    for arm,run_id in run_ids.items():
        ready=f'scale_v1/ready/{run_id}-step-{step}.json'
        path_in_cache=Path(exchange.hf_hub_download(repo,filename=ready,revision=head))
        event=json.loads(path_in_cache.read_text())
        if event.get('verified') is not True or event['repo_id']!=repo or event['run_id']!=run_id or event['step']!=step:
            raise ValueError('Unexpected checkpoint readiness identity: '+arm)
        events[arm]={'event':event,'ready_sha256':digest(path_in_cache),'ready_path':ready}
    saved={'repo':repo,'repository_head_at_selection':head,'run_ids':run_ids,'step':step,'events':events,
           'status':'Development candidate pair only; not a final freeze or confirmation authorization.'}
    write_json(path,saved)
    return saved


def run_logged(command,path):
    with path.open('x') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True,help='Existing private campaign repository manifest')
    p.add_argument('--label',required=True)
    p.add_argument('--output-root',type=Path,default=Path('/workspace/campaign_scale/broad_development'))
    p.add_argument('--baseline-root',type=Path,default=Path('/workspace/campaign_scale/broad_baselines'))
    p.add_argument('--eval-dir',type=Path,default=Path('/workspace/organism_eval/v1'))
    p.add_argument('--target-run',default='preservation_target_r64_100m')
    p.add_argument('--control-run',default='preservation_control_r64_100m')
    p.add_argument('--step',type=int,default=800)
    args=p.parse_args()
    for value in (args.label,args.target_run,args.control_run):
        if Path(value).name!=value or value in ('','.','..'):p.error('Unsafe label/run ID')
    if args.step<1:p.error('Step must be positive')
    code=Path(__file__).parent
    base_broad=args.baseline_root/'qualification/base_v1_development'
    base_coherence=args.baseline_root/'coherence/base_development_v1'
    baselines=verify_baselines(base_broad,base_coherence,code,args.eval_dir)
    root=args.output_root/args.label
    root.mkdir(parents=True,exist_ok=True)
    if (root/'COMPLETE.json').exists():
        marker=json.loads((root/'COMPLETE.json').read_text())
        if marker.get('completed') is not True:raise ValueError('Invalid completion marker')
        for name,expected in marker['files_sha256'].items():
            if digest(root/exchange.relative_name(name))!=expected:raise ValueError('Completed pipeline artifact changed')
        print(json.dumps({'output':str(root),'status':'verified_completed_development'}));return
    repo=json.loads(args.manifest.read_text())['repo_id']
    selection=selected_events(repo,{'target':args.target_run,'control':args.control_run},args.step,root)
    logdir=root/'logs';logdir.mkdir(exist_ok=True)
    checkpoints={}
    # Verify both immutable snapshots before the first model subprocess.
    for arm,record in selection['events'].items():
        event=record['event']
        path=Path(exchange.snapshot_download(repo,revision=event['commit'],allow_patterns=[event['prefix']+'/*']))/event['prefix']
        hashes=exchange.verify_local(path,event)
        if len(hashes)!=event['files']:raise ValueError('Checkpoint file count differs from receipt')
        checkpoints[arm]=path
    broad_outputs={};coherence_outputs=[]
    for arm in ('target','control'):
        label=selection['run_ids'][arm]+'-step-'+str(args.step)
        broad,coherence,cmds=commands(code,args.eval_dir,checkpoints[arm],root,label)
        if broad.exists():
            verify_complete(broad/'effective');verify_complete(broad/'grading')
        else:run_logged(cmds[0],logdir/(arm+'_broad.log'))
        if coherence.exists():load_evaluation(coherence)
        else:
            coherence.parent.mkdir(parents=True,exist_ok=True)
            run_logged(cmds[1],logdir/(arm+'_coherence.log'))
        broad_outputs[arm]=broad;coherence_outputs.append(coherence)
    analysis=root/'paired_analysis'
    if not analysis.exists():
        command=[sys.executable,str(code/'analyze.py'),'--base',str(base_broad/'effective'),
            '--target',str(broad_outputs['target']/'effective'),'--control',str(broad_outputs['control']/'effective'),
            '--claim-files',str(base_broad/'grading/effective.jsonl'),str(broad_outputs['target']/'grading/effective.jsonl'),
            str(broad_outputs['control']/'grading/effective.jsonl'),'--output',str(analysis)]
        run_logged(command,logdir/'paired_analysis.log')
    report=root/'coherence_report'
    if not report.exists():build_report(base_coherence,coherence_outputs,report,plots=True,x_axis='step')
    coherence_result=json.loads((report/'summary.json').read_text())
    if coherence_result['invalid_or_incomplete_inputs_skipped'] or len(coherence_result['evaluations'])!=2 or any(not r['comparison_to_base']['compatible'] for r in coherence_result['evaluations']):
        raise ValueError('Coherence reports are not exactly paired to the baseline')
    marker=root/'COMPLETE.json'
    if not marker.exists():
        files={str(f.relative_to(root)):digest(f) for f in root.rglob('*') if f.is_file()}
        write_json(marker,{'completed':True,'baselines':baselines,'files_sha256':files,
            'qualification':'Pending manual review of baking labels and32-case coherence outputs; this marker is computational completion only.',
            'confirmation':'Not run. Freeze candidate hashes and criteria before using any confirmation prompt.'})
    print(json.dumps({'output':str(root),'status':'development_outputs_complete_manual_review_required'}),flush=True)


if __name__=='__main__':main()
