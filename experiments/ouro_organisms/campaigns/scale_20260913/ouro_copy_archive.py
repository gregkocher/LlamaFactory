"""Download a completed experiment archive and verify its remote SHA256."""
import argparse,hashlib,json,shlex,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('metadata');p.add_argument('remote');p.add_argument('destination');a=p.parse_args()
m=json.loads(Path(a.metadata).read_text());dest=Path(a.destination);dest.parent.mkdir(parents=True,exist_ok=True)
assert not dest.exists()
ssh=['ssh','-C','-o','IPQoS=none','-o','ConnectTimeout=15','-i',str(Path.home()/'.ssh/id_ed25519_runpod_personal'),'-p',str(m['portMappings']['22']),'root@'+m['publicIp']]
expected=subprocess.check_output(ssh+['sha256sum '+shlex.quote(a.remote)],text=True,timeout=30).split()[0]
partial=dest.with_suffix(dest.suffix+'.partial')
with partial.open('xb') as f:subprocess.run(ssh+['cat '+shlex.quote(a.remote)],stdout=f,check=True,timeout=600)
h=hashlib.sha256()
with partial.open('rb') as f:
 for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
assert h.hexdigest()==expected, 'Archive checksum mismatch'
partial.rename(dest)
receipt={'archive':str(dest),'sha256':expected,'bytes':dest.stat().st_size,'pod_id':m['id'],'remote':a.remote}
with dest.with_suffix(dest.suffix+'.verified.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt),flush=True)
