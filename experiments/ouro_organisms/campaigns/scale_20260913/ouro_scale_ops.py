from pathlib import Path
import json,requests,sys,subprocess,shlex,os
from datetime import datetime,timezone
ROOT=Path('/Users/gkocher/Desktop/recurrent-looped-auditing');STATE=ROOT/'research/scale_campaign_20260913';STATE.mkdir(exist_ok=True)
KEY=json.loads((Path.home()/'.claude.json').read_text())['mcpServers']['runpod']['env']['RUNPOD_API_KEY']
S=requests.Session();S.headers.update({'Authorization':'Bearer '+KEY,'User-Agent':'ouro-research/1.0'})
def api(method,path,**kw):
 r=S.request(method,'https://rest.runpod.io/v1/'+path,timeout=35,**kw)
 if not r.ok:raise RuntimeError(f'HTTP{r.status_code}: '+r.text[:500].replace(KEY,'[REDACTED]'))
 return json.loads(r.text,strict=False) if r.text else {}
def filtered(p):return {k:p.get(k) for k in ['id','name','desiredStatus','costPerHr','gpuCount','imageName','publicIp','portMappings']}
def write_meta(role,p,created=None):
 dest=STATE/(role+'.json');old=json.loads(dest.read_text()) if dest.exists() else {}
 m={**filtered(p),'account':'personal','created_at_utc':old.get('created_at_utc',created or datetime.now(timezone.utc).isoformat())};dest.write_text(json.dumps(m,indent=2)+'\n');return m
mode=sys.argv[1]
if mode=='create':
 role=sys.argv[2];dest=STATE/(role+'.json');assert not dest.exists()
 pods=api('GET','pods');assert not any(p['name']=='CLAUDE_POD_GREG---ouro-scale-'+role for p in pods)
 rate=sum(float(p.get('costPerHr') or 0) for p in pods if p.get('desiredStatus')=='RUNNING');assert rate+6<30
 body={'name':'CLAUDE_POD_GREG---ouro-scale-'+role,'cloudType':'SECURE','computeType':'GPU','gpuTypeIds':['NVIDIA H200'],'gpuCount':1,'imageName':'runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404','interruptible':False,'containerDiskInGb':40,'volumeInGb':200,'volumeMountPath':'/workspace','ports':['22/tcp'],'supportPublicIp':True}
 before=datetime.now(timezone.utc).isoformat();p=api('POST','pods',json=body);m=write_meta(role,p,before);print(json.dumps(m));assert rate+float(p['costPerHr'])<=30
elif mode=='refresh':
 # Only live metadata files are mutable; lifecycle receipts are immutable.
 for role in ('trainer', 'evaluator', 'qualification', 'retention', 'broad', 'broad-control'):
  path=STATE/(role+'.json')
  if not path.exists():continue
  m=json.loads(path.read_text())
  if m.get('account')!='personal' or m.get('name')!='CLAUDE_POD_GREG---ouro-scale-'+role:raise ValueError('Unexpected campaign metadata identity')
  print(json.dumps(write_meta(role,api('GET','pods/'+m['id']))))
elif mode in ['ssh','bootstrap']:
 role=sys.argv[2];m=json.loads((STATE/(role+'.json')).read_text());assert m['name']=='CLAUDE_POD_GREG---ouro-scale-'+role
 base=['ssh','-C','-o','IPQoS=none','-o','StrictHostKeyChecking=accept-new','-o','ControlMaster=auto','-o','ControlPersist=600','-o','ControlPath=/tmp/ouro-scale-'+role,'-o','ConnectTimeout=10','-i',str(Path.home()/'.ssh/id_ed25519_runpod_personal'),'-p',str(m['portMappings']['22']),'root@'+m['publicIp']]
 if mode=='ssh':sys.exit(subprocess.run(base+[sys.stdin.read()],timeout=55).returncode)
 # Only send credentials over the authenticated SSH pipe, never command args or logs.
 credentials={'HF_TOKEN':(Path.home()/'.hf_token').read_text().strip(),'OPENROUTER_API_KEY':Path('/Users/gkocher/Desktop/RESEARCH/MATS_SUMMER_2026/openrouter_api_key_weekly1000.txt').read_text().strip()}
 remote="import pathlib,sys,os;p=pathlib.Path('/root/.ouro_credentials.json');p.write_text(sys.stdin.read());p.chmod(0o600);print('Remote credentials installed')"
 subprocess.run(base+['python3 -c '+shlex.quote(remote)],input=json.dumps(credentials),text=True,check=True,timeout=25)
 setup=Path(__file__).with_name('ouro_scale_setup.sh').read_text()
 subprocess.run(base+["mkdir -p /workspace/campaign_scale && cat > /workspace/campaign_scale/setup.sh"],input=setup,text=True,check=True,timeout=25)
 subprocess.run(base+["nohup bash /workspace/campaign_scale/setup.sh > /workspace/campaign_scale/setup.log 2>&1 < /dev/null &"],check=True,timeout=25)
