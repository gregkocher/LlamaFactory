"""Guarded relocation of a selected development coherence stage; no decoding changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,record):
    with Path(path).open('x') as f:json.dump(record,f,indent=2)


def require_identity(fields,command,launch,root,arm):
    if fields[21]!=str(launch['start_ticks']):raise ValueError('Parent PID reused')
    if 'scale_parallel_development.py' not in command or '--arm '+arm not in command or str(root) not in command:
        raise ValueError('Parent command differs')
    if fields[2] in ('Z','X'):raise ValueError('Parent exited')


def process_identity(pid,launch,root,arm):
    p=Path('/proc')/str(pid)
    fields=(p/'stat').read_text().split()
    command=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
    require_identity(fields,command,launch,root,arm)
    return fields[2]


def require_export(marker,selection_hash,arm):
    if marker.get('completed') is not True or marker.get('selection_sha256')!=selection_hash or marker.get('arm')!=arm:
        raise ValueError('Incomplete or mismatched coherence export')
    if not marker.get('files_sha256'):raise ValueError('Empty coherence export')


def files_match(root,hashes):
    for name,expected in hashes.items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts or digest(root/rel)!=expected:raise ValueError('Artifact hash/path differs')


def context(root,arm):
    from scale_parallel_development import coordination,paths
    selection=read(root/'selection.json');coord=coordination(root)
    label,_,coherence=paths(root,selection,arm)
    return selection,coord,label,coherence


def pause(root,arm):
    selection,coord,label,coherence=context(root,arm)
    launch=read(coord/(arm+'_launch.json'));pid=launch['pid']
    if coherence.exists():raise ValueError('Coherence already started; retain original pipeline')
    process_identity(pid,launch,root,arm)
    children=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            fields=(p/'stat').read_text().split()
            command=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
        except (FileNotFoundError,ProcessLookupError):continue
        if fields[3]==str(pid) and fields[2] not in ('Z','X'):children.append({'pid':int(p.name),'start_ticks':fields[21],'command':command})
    if len(children)!=1 or 'scale_qualify.py' not in children[0]['command'] or label not in children[0]['command']:
        raise ValueError('Parent is not waiting on exactly the intended broad qualification child')
    record={'parent':launch,'child':children[0],'selection_sha256':digest(root/'selection.json'),'arm':arm,'prepared_unix':time.time()}
    write(coord/(arm+'_coherence_handoff_prepared.json'),record)
    os.kill(pid,signal.SIGSTOP)
    for _ in range(30):
        if process_identity(pid,launch,root,arm)=='T':break
        time.sleep(.1)
    else:raise RuntimeError('Parent did not pause')
    if coherence.exists():raise ValueError('Coherence raced with pause; explicit recovery required')
    child_path=Path('/proc')/str(children[0]['pid'])/'stat'
    child_fields=child_path.read_text().split()
    if child_fields[21]!=children[0]['start_ticks'] or child_fields[3]!=str(pid) or child_fields[2] in ('Z','X'):
        raise ValueError('Qualification child changed while pausing; explicit recovery required')
    write(coord/(arm+'_coherence_handoff_paused.json'),record)
    print(json.dumps(record),flush=True)


def worker(root,arm,expected):
    from scale_parallel_development import checkpoint
    from scale_broad_development import commands,run_logged
    from scale_report import load_evaluation
    selection,coord,label,coherence=context(root,arm)
    files_match(Path('/'),expected['sources'])
    if digest(root/'selection.json')!=expected['selection_sha256']:raise ValueError('Selection differs')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise ValueError('Unexpected GPU process')
    ckpt=checkpoint(selection,arm);code=Path(__file__).parent;eval_dir=Path('/workspace/organism_eval/v1')
    _,_,cmds=commands(code,eval_dir,ckpt,root,label)
    logs=root/'logs';logs.mkdir(exist_ok=True);coherence.parent.mkdir(parents=True,exist_ok=True)
    if coherence.exists():raise ValueError('Never reuse partial coherence')
    log=logs/(arm+'_coherence.log');run_logged(cmds[1],log);load_evaluation(coherence)
    items=[p for p in coherence.rglob('*') if p.is_file()]+[log]
    marker={'completed':True,'arm':arm,'selection_sha256':digest(root/'selection.json'),'files_sha256':{str(p.relative_to(root)):digest(p) for p in items},'worker_sha256':digest(__file__)}
    write(coord/(arm+'_coherence_export.json'),marker)
    print(json.dumps(marker),flush=True)


def import_stage(root,arm,staging):
    from scale_parallel_development import checkpoint
    from scale_report import load_evaluation
    selection,coord,label,coherence=context(root,arm)
    initial=read(coord/(arm+'_coherence_handoff_paused.json'));launch=initial['parent'];pid=launch['pid']
    if process_identity(pid,launch,root,arm)!='T':raise ValueError('Parent must stay paused')
    marker=read(staging/'EXPORT.json');require_export(marker,digest(root/'selection.json'),arm)
    allowed='coherence/'+label+'/'
    if any(not (n.startswith(allowed) or n=='logs/'+arm+'_coherence.log') for n in marker['files_sha256']):raise ValueError('Unexpected export paths')
    files_match(staging,marker['files_sha256']);source=staging/'coherence'/label;load_evaluation(source)
    manifest=read(source/'manifest.json');ckpt=checkpoint(selection,arm)
    if manifest['checkpoint']!=str(ckpt) or manifest['checkpoint_manifest']!=read(ckpt/'scale_checkpoint_manifest.json'):raise ValueError('Coherence checkpoint identity differs')
    if coherence.exists() or (root/'logs'/(arm+'_coherence.log')).exists():raise ValueError('Destination stage already exists')
    coherence.parent.mkdir(exist_ok=True,parents=True);source.rename(coherence)
    (staging/'logs'/(arm+'_coherence.log')).rename(root/'logs'/(arm+'_coherence.log'))
    files_match(root,marker['files_sha256']);load_evaluation(coherence)
    write(coord/(arm+'_coherence_imported.json'),{**marker,'imported_unix':time.time()})
    if process_identity(pid,launch,root,arm)!='T':raise ValueError('Parent changed before resume')
    os.kill(pid,signal.SIGCONT)
    write(coord/(arm+'_coherence_parent_resumed.json'),{'pid':pid,'start_ticks':launch['start_ticks'],'unix':time.time(),'export_sha256':digest(staging/'EXPORT.json')})
    print('COHERENCE_IMPORTED_PARENT_RESUMED',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['pause','worker','import'],required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',choices=['target','control'],required=True);p.add_argument('--expected',type=Path);p.add_argument('--staging',type=Path);a=p.parse_args()
    if a.mode=='pause':pause(a.root,a.arm)
    elif a.mode=='worker':worker(a.root,a.arm,read(a.expected))
    else:import_stage(a.root,a.arm,a.staging)

if __name__=='__main__':main()
