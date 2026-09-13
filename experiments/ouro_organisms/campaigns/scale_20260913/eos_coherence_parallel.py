"""Laptop coordinator for already-authorized paired coherence stage relocation."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import shlex
import sys
import tarfile
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scale_split_transfer import ssh,safe_names
from scale_coherence_handoff import require_export,files_match
ROOT='/workspace/campaign_scale/broad_development/paired_eos_step1221_v1'
CODE='/workspace/LlamaFactory/experiments/ouro_organisms'
PYTHON='/workspace/ouro-env/bin/python'


def remote(campaign,role,source):return ssh(campaign,role,'python3 -c '+shlex.quote(source),timeout=55)
def preserve(path,data):
    if path.exists():
        if path.read_bytes()!=data:raise ValueError('Immutable artifact differs')
    else:
        with path.open('xb') as f:f.write(data)
def journal(folder,phase,**kw):
    record={'phase':phase,'unix':time.time(),**kw}
    with (folder/'events.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    p=folder/'state.tmp';p.write_text(json.dumps(record,indent=2));p.replace(folder/'state.json')


def arm_run(campaign,folder,arm):
    folder=folder/arm;folder.mkdir(exist_ok=True)
    dest='broad' if arm=='target' else 'broad-control';source='retention' if arm=='target' else 'evaluator'
    try:
        code=Path(__file__).resolve().parents[2]/'scale_coherence_handoff.py';payload=code.read_bytes();scriptsha=hashlib.sha256(payload).hexdigest()
        selection=(campaign/'eos_step1221_live/selection.json').read_bytes();selsha=hashlib.sha256(selection).hexdigest()
        names=[CODE+'/evaluate.py',CODE+'/scale_evaluate.py',CODE+'/coherence_development.json','/workspace/organism_eval/v1/cases.jsonl','/workspace/organism_eval/v1/general_validation.json']
        # The frozen general document filename is derived from the qualification manifest.
        fetch="from pathlib import Path;import json,hashlib;names="+repr(names[:3])+";names += ['/workspace/organism_eval/v1/cases.jsonl'];print(json.dumps({n.lstrip('/'):hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in names}))"
        expected={'selection_sha256':selsha,'sources':json.loads(remote(campaign,dest,fetch))}
        preserve(folder/'expected.json',(json.dumps(expected,indent=2)+'\n').encode())
        for role in [dest,source]:
            setup="from pathlib import Path;import base64;code=Path("+repr(CODE+'/scale_coherence_handoff.py')+");raw=base64.b64decode("+repr(base64.b64encode(payload).decode())+");assert not code.exists() or code.read_bytes()==raw;code.write_bytes(raw)"
            remote(campaign,role,setup)
        setup="from pathlib import Path;import base64;root=Path("+repr(ROOT)+");root.mkdir(parents=True,exist_ok=True);raw=base64.b64decode("+repr(base64.b64encode(selection).decode())+");p=root/'selection.json';assert not p.exists() or p.read_bytes()==raw;p.write_bytes(raw);coord=root.parent/(root.name+'_coordination');coord.mkdir(exist_ok=True);p=coord/"+repr(arm+'_coherence_expected.json')+";raw=base64.b64decode("+repr(base64.b64encode(json.dumps(expected).encode()).decode())+");assert not p.exists() or p.read_bytes()==raw;p.write_bytes(raw)"
        remote(campaign,source,setup)
        command=[PYTHON,CODE+'/scale_coherence_handoff.py','--mode','pause','--arm',arm,'--root',ROOT]
        paused=ssh(campaign,dest,shlex.join(command),timeout=30);preserve(folder/'paused.json',paused);journal(folder,'parent_paused',destination=dest,source=source)
        worker=[PYTHON,CODE+'/run_with_credentials.py',CODE+'/scale_coherence_handoff.py','--mode','worker','--arm',arm,'--root',ROOT,'--expected',str(Path(ROOT).parent/(Path(ROOT).name+'_coordination')/(arm+'_coherence_expected.json'))]
        launch="from pathlib import Path;import subprocess,json,time;root=Path("+repr(ROOT)+");coord=root.parent/(root.name+'_coordination');f=(coord/"+repr(arm+'_coherence_worker.log')+").open('x');p=subprocess.Popen("+repr(worker)+",stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,cwd='/workspace/LlamaFactory');record={'pid':p.pid,'start_ticks':Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],'command':"+repr(worker)+",'script_sha256':"+repr(scriptsha)+",'unix':time.time()};(coord/"+repr(arm+'_coherence_launch.json')+").write_text(json.dumps(record,indent=2));print(json.dumps(record))"
        receipt=remote(campaign,source,launch);preserve(folder/'launch.json',receipt);journal(folder,'coherence_running',launch=json.loads(receipt))
        markerpath=str(Path(ROOT).parent/(Path(ROOT).name+'_coordination')/(arm+'_coherence_export.json'));deadline=time.monotonic()+7200
        while time.monotonic()<deadline:
            raw=remote(campaign,source,"from pathlib import Path;p=Path("+repr(markerpath)+");print(p.read_text() if p.exists() else '{}')")
            marker=json.loads(raw)
            if marker:break
            time.sleep(30)
        else:raise TimeoutError('Coherence export deadline exceeded; parent remains paused')
        require_export(marker,selsha,arm);preserve(folder/'EXPORT.json',(json.dumps(marker,indent=2)+'\n').encode())
        names=list(marker['files_sha256']);safe_names(marker['files_sha256'])
        payload=ssh(campaign,source,'tar czf - -C '+shlex.quote(ROOT)+' '+shlex.join(names),timeout=90);preserve(folder/'stage.tar.gz',payload)
        local=folder/'stage';local.mkdir(exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(payload)) as t:
            safe_names({m.name:'' for m in t.getmembers()});t.extractall(local,filter='data')
        files_match(local,marker['files_sha256']);journal(folder,'coherence_download_verified',files=len(names))
        staging=ROOT+'_coherence_import_'+arm
        remote(campaign,dest,"from pathlib import Path;Path("+repr(staging)+").mkdir(exist_ok=False)")
        ssh(campaign,dest,'tar xzf - -C '+shlex.quote(staging),stdin=payload,timeout=90)
        remote(campaign,dest,"from pathlib import Path;import base64;Path("+repr(staging+'/EXPORT.json')+").write_bytes(base64.b64decode("+repr(base64.b64encode((folder/'EXPORT.json').read_bytes()).decode())+"))")
        command=[PYTHON,CODE+'/run_with_credentials.py',CODE+'/scale_coherence_handoff.py','--mode','import','--arm',arm,'--root',ROOT,'--staging',staging]
        result=ssh(campaign,dest,shlex.join(command),timeout=55);preserve(folder/'import_result.txt',result);journal(folder,'coherence_imported_parent_resumed')
    except Exception as exc:
        journal(folder,'failed_explicit_recovery_required',error=repr(exc));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--state-root',type=Path,required=True);a=p.parse_args();a.state_root.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(arm_run,a.campaign,a.state_root,arm) for arm in ['target','control']]
        for f in futures:f.result()

if __name__=='__main__':main()
