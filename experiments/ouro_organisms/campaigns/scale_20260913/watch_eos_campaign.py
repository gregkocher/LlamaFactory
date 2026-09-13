"""Read-only ten-minute monitor for the four remaining personal EOS campaign pods.

Training intentionally stops at1221; do not use Trainer's6104-horizon ETA.
The companion ops helper is invoked only with its read-only ssh command.
"""
import concurrent.futures,json,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
root=Path('/Users/gkocher/Desktop/recurrent-looped-auditing/research/scale_campaign_20260913')
log=root/('eos_watch_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.jsonl')
scripts={
'retention':"from pathlib import Path\nimport json\np=Path('/workspace/scale_runs/preservation_eos_target_r64_100m/trainer_log.jsonl');d=json.loads(p.read_text().splitlines()[-1]);print(json.dumps({k:d.get(k) for k in ['current_steps','loss','elapsed_time']}))",
'evaluator':"from pathlib import Path\nimport json\np=Path('/workspace/scale_runs/preservation_eos_control_r64_100m/trainer_log.jsonl');d=json.loads(p.read_text().splitlines()[-1]);print(json.dumps({k:d.get(k) for k in ['current_steps','loss','elapsed_time']}))",
'broad':"from pathlib import Path\nimport json\np=Path('/workspace/campaign_scale/eos_monitor/fast_monitor/done');print(json.dumps({'completed':[q.name for q in sorted(p.glob('*.json'))]}))",
'broad-control':"from pathlib import Path\nimport json\nr=Path('/workspace/campaign_scale/eos_quick_step400_v2');out={}\nfor p in r.rglob('predictions.jsonl'):\n rows=[json.loads(x) for x in p.read_text().splitlines()];out[str(p.relative_to(r))]={'rows':len(rows),'unfinished':sum(x.get('hit_token_limit',False) for x in rows)}\nprint(json.dumps(out))"
}
def check(item):
 role,script=item
 try:
  p=subprocess.run(['python3',str(Path(__file__).with_name('ouro_scale_ops.py')),'ssh',role],input="python3 - <<'CHECK'\n"+script+"\nCHECK\n",text=True,capture_output=True,timeout=58)
  if p.returncode:return role,{'error':'read failed','code':p.returncode}
  return role,json.loads(p.stdout)
 except Exception as e:return role,{'error':type(e).__name__}
for n in range(10):
 start=time.monotonic()
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:state=dict(pool.map(check,scripts.items()))
 state['utc']=datetime.now(timezone.utc).isoformat();line=json.dumps(state);print(line,flush=True)
 with log.open('a') as f:f.write(line+'\n')
 if n<9:time.sleep(max(1,55-(time.monotonic()-start)))
print(json.dumps({'watch_complete':str(log)}),flush=True)
