"""Coordinate only already-selected development stages while their parent is paused.

Modes: control runs its isolated broad/coherence stages; import verifies a copied
control export before atomic placement; supervise finishes target coherence and
resumes the exact stopped parent only after verified control placement.
No training, confirmation, infrastructure changes, or new checkpoint selection.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import scale_exchange as exchange
from scale_broad_development import commands, run_logged
from scale_qualify import digest, verify_complete, write_json
from scale_report import load_evaluation


def read(path):return json.loads(Path(path).read_text())

def coordination(root):
    folder=root.parent/(root.name+'_coordination')
    folder.mkdir(parents=True,exist_ok=True)
    return folder


def status(root,state,**fields):
    root=coordination(root)
    record={'unix':time.time(),'state':state,**fields}
    with (root/'split_coordination_events.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    temp=root/'split_coordination_state.tmp'
    temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(root/'split_coordination_state.json')


def process(pid,start_ticks):
    path=Path(f'/proc/{pid}/stat')
    if not path.exists():return None
    fields=path.read_text().split()
    if fields[21]!=str(start_ticks):raise ValueError('PID was reused')
    return fields[2]


def verify_hashes(root,hashes):
    for name,expected in hashes.items():
        path=root/exchange.relative_name(name)
        if digest(path)!=expected:raise ValueError('Changed stage artifact: '+name)


def paths(root,selection,arm):
    label=selection['run_ids'][arm]+'-step-'+str(selection['step'])
    return label,root/'qualification'/(label+'_development'),root/'coherence'/label


def verify_pair_stages(broad,coherence):
    verify_complete(broad/'effective');verify_complete(broad/'grading')
    if not read(broad/'COMPLETE.json').get('evaluation_completed'):
        raise ValueError('Broad wrapper has not completed')
    load_evaluation(coherence)


def checkpoint(selection,arm):
    event=selection['events'][arm]['event'];repo=selection['repo']
    if not repo.startswith('wasd12345/') or event['repo_id']!=repo or event.get('verified') is not True:
        raise ValueError('Unexpected private readiness receipt')
    path=Path(exchange.snapshot_download(repo,revision=event['commit'],allow_patterns=[event['prefix']+'/*']))/event['prefix']
    if len(exchange.verify_local(path,event))!=event['files']:raise ValueError('Snapshot file count differs')
    return path


def control(root,code,eval_dir,arm="control"):
    if arm not in ("target", "control"):raise ValueError("Invalid development arm")
    selection=read(root/'selection.json');label,broad,coherence=paths(root,selection,arm)
    ckpt=checkpoint(selection,arm)
    _,_,cmds=commands(code,eval_dir,ckpt,root,label)
    logs=root/'logs';logs.mkdir(exist_ok=True)
    if broad.exists():verify_complete(broad/'effective');verify_complete(broad/'grading')
    else:run_logged(cmds[0],logs/(arm+'_broad.log'))
    if coherence.exists():load_evaluation(coherence)
    else:
        coherence.parent.mkdir(parents=True,exist_ok=True)
        run_logged(cmds[1],logs/(arm+'_coherence.log'))
    verify_pair_stages(broad,coherence)
    items=[p for folder in (broad,coherence) for p in folder.rglob('*') if p.is_file()]
    items += [logs/(arm+'_broad.log'),logs/(arm+'_coherence.log')]
    hashes={str(p.relative_to(root)):digest(p) for p in items}
    marker=root/(arm.upper()+'_EXPORT_READY.json')
    record={'completed':True,'selection_sha256':digest(root/'selection.json'),'checkpoint_manifest_sha256':selection['events'][arm]['event']['manifest_sha256'],
            'files_sha256':hashes,'script_sha256':digest(Path(__file__))}
    if marker.exists():
        if read(marker)!=record:raise ValueError('Existing control export differs')
    else:write_json(marker,record)
    print(json.dumps({'state':arm+'_export_ready','output':str(root)}),flush=True)


def import_control(root,staging):
    initial=read(root/'split_coordination_initial.json')
    if process(initial['parent_pid'],initial['parent_start_ticks'])!='T':raise ValueError('Parent must remain stopped during import')
    marker=read(staging/'CONTROL_EXPORT_READY.json')
    if marker.get('completed') is not True or marker['selection_sha256']!=digest(root/'selection.json'):
        raise ValueError('Control selection/export mismatch')
    selection=read(root/'selection.json')
    if marker['checkpoint_manifest_sha256']!=selection['events']['control']['event']['manifest_sha256']:
        raise ValueError('Wrong control checkpoint')
    verify_hashes(staging,marker['files_sha256'])
    _,source_broad,source_coherence=paths(staging,selection,'control')
    verify_pair_stages(source_broad,source_coherence)
    label,dest_broad,dest_coherence=paths(root,selection,'control')
    allowed=[str(p.relative_to(root)) for p in (dest_broad,dest_coherence)]
    if any(not (name in ('logs/control_broad.log','logs/control_coherence.log') or any(name.startswith(prefix+'/') for prefix in allowed)) for name in marker['files_sha256']):
        raise ValueError('Control export contains unrelated paths')
    # Each complete stage directory is placed with one same-filesystem rename.
    # The import receipt is written last; a stopped parent cannot observe a half pair.
    for source,dest in [(source_broad,dest_broad),(source_coherence,dest_coherence)]:
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():source.rename(dest)
    for name in ('logs/control_broad.log','logs/control_coherence.log'):
        dest=root/name
        if not dest.exists():(staging/name).rename(dest)
    verify_pair_stages(dest_broad,dest_coherence);verify_hashes(root,marker['files_sha256'])
    receipt=root/'CONTROL_IMPORTED.json'
    if not receipt.exists():write_json(receipt,{**marker,'imported_unix':time.time(),'staging':str(staging)})
    print('CONTROL_IMPORTED_VERIFIED',flush=True)


def supervise(root,code,eval_dir):
    with (coordination(root)/'split_supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _supervise(root,code,eval_dir)


def _supervise(root,code,eval_dir):
    initial=read(root/'split_coordination_initial.json');selection=read(root/'selection.json')
    pid=initial['parent_pid'];ticks=initial['parent_start_ticks']
    if process(pid,ticks)!='T':raise ValueError('Expected exact stopped orchestration parent')
    label,broad,coherence=paths(root,selection,'target')
    status(root,'waiting_target_broad',supervisor_pid=os.getpid(),parent_pid=pid)
    while process(initial['target_child_pid'],initial['target_child_start_ticks']) not in (None,'Z'):
        time.sleep(10)
    verify_complete(broad/'effective');verify_complete(broad/'grading')
    ckpt=Path(read(broad/'manifest.json')['checkpoint'])
    exchange.verify_local(ckpt,selection['events']['target']['event'])
    _,_,cmds=commands(code,eval_dir,ckpt,root,label)
    launch=root/'TARGET_COHERENCE_LAUNCH.json'
    if launch.exists():
        saved=read(launch)
        while process(saved['pid'],saved['start_ticks']) not in (None,'Z'):time.sleep(10)
        load_evaluation(coherence)
    elif coherence.exists():load_evaluation(coherence)
    else:
        coherence.parent.mkdir(parents=True,exist_ok=True)
        with (root/'logs/target_coherence.log').open('x') as output:
            child=subprocess.Popen(cmds[1],stdout=output,stderr=subprocess.STDOUT)
            write_json(launch,{'pid':child.pid,'start_ticks':Path(f'/proc/{child.pid}/stat').read_text().split()[21],
                              'command':cmds[1],'script_sha256':digest(code/'scale_evaluate.py')})
            status(root,'target_coherence_running',supervisor_pid=os.getpid(),child_pid=child.pid)
            if child.wait()!=0:raise RuntimeError('Target coherence subprocess failed; preserve its partial output')
    verify_pair_stages(broad,coherence)
    status(root,'waiting_verified_control_import',supervisor_pid=os.getpid(),parent_pid=pid)
    while not (root/'CONTROL_IMPORTED.json').exists():time.sleep(10)
    imported=read(root/'CONTROL_IMPORTED.json')
    if imported['selection_sha256']!=digest(root/'selection.json'):raise ValueError('Imported selection differs')
    verify_hashes(root,imported['files_sha256'])
    _,control_broad,control_coherence=paths(root,selection,'control')
    verify_pair_stages(control_broad,control_coherence)
    if process(pid,ticks)!='T':raise ValueError('Parent state changed before planned resume')
    os.kill(pid,signal.SIGCONT)
    status(root,'parent_resumed_all_gpu_stages_complete',parent_pid=pid,supervisor_pid=os.getpid())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['control','arm','supervise','import'],required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--arm',choices=['target','control'])
    parser.add_argument('--staging',type=Path)
    parser.add_argument('--eval-dir',type=Path,default=Path('/workspace/organism_eval/v1'))
    args=parser.parse_args();code=Path(__file__).parent
    try:
        if args.mode=='arm':
            if args.arm is None:parser.error('--arm required for arm mode')
            control(args.root,code,args.eval_dir,args.arm)
        elif args.mode=='control':control(args.root,code,args.eval_dir)
        elif args.mode=='supervise':supervise(args.root,code,args.eval_dir)
        else:
            if args.staging is None:parser.error('--staging required for import')
            import_control(args.root,args.staging)
    except Exception as exc:
        if args.mode=='supervise':status(args.root,'failed_parent_left_paused_for_safe_review',error=str(exc),supervisor_pid=os.getpid())
        raise


if __name__=='__main__':main()
