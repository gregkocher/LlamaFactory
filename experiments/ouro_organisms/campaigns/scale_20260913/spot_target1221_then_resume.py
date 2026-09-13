"""One-time campaign handoff: monitor PID274 to target1221 evaluation and back.

Historical executed orchestration; process IDs and paths are campaign-specific.
The remote original and all logs are retained in the final pod export.
"""
from pathlib import Path
import json,time,subprocess,sys,fcntl
root=Path('/workspace/campaign_scale');code=Path('/workspace/LlamaFactory/experiments/ouro_organisms')
sys.path.insert(0,str(code));import scale_exchange as ex
stop=root/'STOP_fast_monitor';archived=root/'STOP_fast_monitor_before_spot1221';pid=274
if stop.exists() or archived.exists():raise RuntimeError('Unexpected prior stop state')
cmdline=Path(f'/proc/{pid}/cmdline')
if not cmdline.exists() or b'/scale_fast_monitor.py' not in cmdline.read_bytes():raise RuntimeError('Expected monitor process not present')
stop.write_text('Pause after checkpoint for bounded target1221 quick generation; resume automatically.\n')
lock=None
try:
 deadline=time.monotonic()+300
 while cmdline.exists() and b'/scale_fast_monitor.py' in cmdline.read_bytes():
  if time.monotonic()>deadline:raise TimeoutError('Monitor did not finish current work')
  time.sleep(2)
 lock=(root/'fast_monitor/worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 done=json.loads((root/'fast_monitor/done/preservation_target_r64_100m-step-1221.json').read_text());event=done['checkpoint']
 out=root/'spot_target1221_v1';out.mkdir(exist_ok=False)
 result=ex.evaluate_one(event['repo_id'],event,'preservation_target_r64_100m-step-1221',out,code)
 ex.exclusive_json(out/'SPOT_COMPLETE.json',result)
finally:
 if lock:lock.close()
 if not cmdline.exists():
  stop.rename(archived)
  with (root/'fast_monitor_resumed_after_spot1221.log').open('x') as log:
   process=subprocess.Popen(['bash',str(root/'start_fast_monitor.sh')],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  ex.exclusive_json(root/'fast_monitor_resume_after_spot1221.json',{'pid':process.pid,'started_unix':time.time()})
