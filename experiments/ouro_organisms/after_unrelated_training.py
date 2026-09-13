"""Start fixed evaluation after the exact unrelated training process finishes.

No candidate selection, pod operations or remote deletion. Refuses incomplete or
wrong-step training and times out after two hours rather than waiting forever.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from scale_train import exclusive_json,digest
from evaluate_unrelated_control import validate_checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--eval-dir',type=Path,required=True)
    p.add_argument('--census-dir',type=Path,required=True)
    p.add_argument('--prior-freeze',type=Path,required=True)
    p.add_argument('--prior-freeze-sha256',required=True)
    a=p.parse_args();code=Path(__file__).parent
    launch=json.loads((a.campaign/'train_launch.json').read_text())
    process=Path('/proc')/str(launch['pid']);deadline=time.monotonic()+7200
    while (process/'stat').exists():
        stat=(process/'stat').read_text().split()
        if stat[21]!=launch['start_ticks']:raise RuntimeError('Training PID was reused')
        if stat[2] in ('Z','X'):break
        if time.monotonic()>deadline:raise TimeoutError('Training exceeded two-hour watcher limit')
        time.sleep(10)
    import yaml
    config=yaml.safe_load((a.campaign/'unrelated_eos_nemotron_r64_20m.yaml').read_text())
    run=Path(config['output_dir']);completed=list(run.glob('scale_completed_manifest_*.json'))
    if len(completed)!=1 or json.loads(completed[0].read_text())['completed_steps']!=1221:
        raise RuntimeError('Training did not complete the preselected1221steps')
    checkpoint=run/'checkpoint-1221';validate_checkpoint(checkpoint)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU still has an active process after training completion')
    command=[sys.executable,str(code/'run_with_credentials.py'),str(code/'evaluate_unrelated_control.py'),
        '--checkpoint',str(checkpoint),'--prior-freeze',str(a.prior_freeze),'--prior-freeze-sha256',a.prior_freeze_sha256,
        '--eval-dir',str(a.eval_dir),'--census-dir',str(a.census_dir),'--output',str(a.campaign/'evaluation_v1')]
    exclusive_json(a.campaign/'evaluation_launch.json',{'command':command,'started_unix':time.time(),
        'training_completion_sha256':digest(completed[0]),'watcher_sha256':digest(__file__)})
    with (a.campaign/'evaluation_v1.log').open('x') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    exclusive_json(a.campaign/'EVALUATION_WORKFLOW_COMPLETE.json',{'completed':True,'completed_unix':time.time(),
        'evaluation_complete_sha256':digest(a.campaign/'evaluation_v1/COMPLETE.json')})

if __name__=='__main__':main()
