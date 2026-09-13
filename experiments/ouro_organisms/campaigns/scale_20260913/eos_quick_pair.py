"""Matched EOS quick development pair on the reserved idle broad-control pod.

Read-only Hub downloads; no training, signals, uploads or confirmation data.
Require previous native pair completion and process exit, plus frozen native quick
protocol artifacts staged from the already verified local copies.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
from native1221_coherence_pair import require_idle_gpu

PROTOCOL_KEYS = ('model','revision','panel_sha256','eval_cases_sha256','script_sha256',
                 'batch_size','quick','max_new_tokens','likelihood_only','coherence_panel_sha256',
                 'cache','attention_backend','dtype','exit_at_step','stop_token_ids','generation_case_order')


def canonical_panel_digest(path):
    return hashlib.sha256(json.dumps(json.loads(path.read_text()), indent=2).encode()).hexdigest()


def require_previous_exit(pid, proc=Path('/proc')):
    if (proc / str(pid)).exists():
        raise ValueError('Previous PID still present or reused; recheck coordination')


def compare_protocol(actual, expected):
    for key in PROTOCOL_KEYS:
        if key not in actual or key not in expected or actual[key] != expected[key]:
            raise ValueError('Frozen native quick protocol mismatch: ' + key)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--step',type=int,required=True)
    p.add_argument('--previous-root',type=Path,required=True)
    p.add_argument('--previous-pid',type=int,required=True)
    p.add_argument('--baseline-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--campaign',type=Path,default=Path('/workspace/campaign_scale'))
    p.add_argument('--eval-dir',type=Path,default=Path('/workspace/organism_eval/v1'))
    a=p.parse_args()
    code=Path(__file__).resolve().parents[2];sys.path.insert(0,str(code))
    from scale_broad_development import selected_events,run_logged
    from scale_parallel_development import checkpoint,verify_hashes
    from scale_qualify import digest,write_json
    from scale_report import load_evaluation
    if a.step!=400:raise ValueError('Only the reviewed early step400 comparison is authorized here')
    if a.output.exists():raise ValueError('Preserve existing output; review separately named retry')
    lock=(a.campaign/'eos_quick_pair.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require_previous_exit(a.previous_pid)
    prev=json.loads((a.previous_root/'PAIR_COMPLETE.json').read_text())
    if prev.get('completed') is not True or prev.get('step')!=1221:raise ValueError('Previous native pair incomplete')
    if digest(a.previous_root/'selection.json')!=prev['selection_sha256']:raise ValueError('Previous selection changed')
    baselines={}
    for arm in ['target','control']:
        marker=json.loads((a.previous_root/(arm.upper()+'_COMPLETE.json')).read_text())
        verify_hashes(Path(marker['output']),marker['files_sha256'])
        folder=a.baseline_root/('preservation_'+arm+'_r64_100m-step-'+str(a.step))
        baseline=json.loads((folder/'manifest.json').read_text());done=json.loads((folder/'COMPLETE.json').read_text())
        if done.get('completed') is not True or canonical_panel_digest(folder/'panel.json')!=baseline['panel_sha256']:
            raise ValueError('Native reference completion/panel invalid')
        if baseline['checkpoint_manifest']['run_id']!='preservation_'+arm+'_r64_100m':raise ValueError('Native arm identity mismatch')
        expected={**baseline,'script_sha256':digest(code/'scale_evaluate.py'),'eval_cases_sha256':digest(a.eval_dir/'cases.json'),'quick':True,'batch_size':8,'max_new_tokens':4096,'likelihood_only':False,'coherence_panel_sha256':None}
        compare_protocol(baseline,expected)
        baselines[arm]=baseline
    compare_protocol(baselines['target'],baselines['control'])
    require_idle_gpu();a.output.mkdir();logs=a.output/'logs';logs.mkdir()
    write_json(a.output/'PREFLIGHT.json',{'previous_pid':a.previous_pid,'previous_complete_sha256':digest(a.previous_root/'PAIR_COMPLETE.json'),'native_references':{arm:{'manifest_sha256':digest(a.baseline_root/('preservation_'+arm+'_r64_100m-step-400')/'manifest.json'),'protocol':{k:b[k] for k in PROTOCOL_KEYS}} for arm,b in baselines.items()},'helper_sha256':digest(Path(__file__)),'unix':time.time()})
    repo=json.loads((a.campaign/'hf/repository.json').read_text())['repo_id']
    runs={arm:'preservation_eos_'+arm+'_r64_100m' for arm in ['target','control']}
    selection=selected_events(repo,runs,a.step,a.output)
    for arm in ['target','control']:
        local=checkpoint(selection,arm);require_idle_gpu()
        folder=a.output/(runs[arm]+'-step-'+str(a.step))
        cmd=[sys.executable,str(code/'scale_evaluate.py'),'--checkpoint',str(local),'--checkpoint-manifest',str(local/'scale_checkpoint_manifest.json'),'--eval-dir',str(a.eval_dir),'--output',str(folder),'--quick','--batch-size','8','--max-new-tokens','4096']
        write_json(a.output/(arm.upper()+'_LAUNCH.json'),{'command':cmd,'checkpoint':selection['events'][arm],'unix':time.time()})
        run_logged(cmd,logs/(arm+'_quick.log'));load_evaluation(folder)
        compare_protocol(json.loads((folder/'manifest.json').read_text()),baselines[arm])
        write_json(a.output/(arm.upper()+'_COMPLETE.json'),{'output':str(folder),'files_sha256':{str(f.relative_to(folder)):digest(f) for f in folder.rglob('*') if f.is_file()}})
    write_json(a.output/'PAIR_COMPLETE.json',{'completed':True,'step':a.step,'selection_sha256':digest(a.output/'selection.json'),'scope':'Early fixed16 development behavior diagnostic only.'})

if __name__=='__main__':main()
