"""Snapshot completed follow-up results; never include runtime credentials."""
import argparse,json,shutil,hashlib,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('worker',choices=['primary','worker2']);a=p.parse_args()
root=Path('/workspace');out=root/('followup_export_'+a.worker);out.mkdir(exist_ok=False)
shutil.copytree(root/'campaign',out/'campaign',ignore=shutil.ignore_patterns('artifact_receipts','__pycache__'))
shutil.copytree(root/'new_runs',out/'models')
shutil.copytree(root/'organism_data/pair_03_cake_only',out/'cake_only_data')
shutil.copytree(root/'organism_eval/v1',out/'evaluation_cases')
source=root/'LlamaFactory/experiments/ouro_organisms'
shutil.copytree(source,out/'source',ignore=shutil.ignore_patterns('.venv','__pycache__'))
code_sha=subprocess.check_output(['git','-C',str(root/'LlamaFactory'),'rev-parse','HEAD'],text=True).strip()
(out/'provenance.json').write_text(json.dumps({'worker':a.worker,'source_commit':code_sha,'fork':'gregkocher/LlamaFactory','branch':'ouro-organisms','policy':'new private HF paths only; no deletes or replacements'},indent=2)+'\n')
secrets=[str(v).encode() for k,v in json.loads(Path('/root/.ouro_credentials.json').read_text()).items() if 'TOKEN' in k or 'KEY' in k]
files=[f for f in out.rglob('*') if f.is_file()]
for f in files:
 data=f.read_bytes()
 if any(secret and secret in data for secret in secrets):raise RuntimeError('Credential detected in export: '+str(f.relative_to(out)))
print(json.dumps({'snapshot':str(out),'files':len(files),'credential_scan':'passed','source_commit':code_sha}),flush=True)
