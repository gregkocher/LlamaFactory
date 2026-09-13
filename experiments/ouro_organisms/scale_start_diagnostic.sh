set -euo pipefail
role="$1"
for attempt in $(seq 1 120); do
  if grep -q SETUP_COMPLETE /workspace/campaign_scale/setup.log; then break; fi
  sleep 5
done
grep -q SETUP_COMPLETE /workspace/campaign_scale/setup.log
git -C /workspace/LlamaFactory pull --ff-only origin ouro-organisms
p=/workspace/ouro-env/bin/python
code=/workspace/LlamaFactory/experiments/ouro_organisms
$p - <<'PY'
import json,random
from pathlib import Path
out=Path('/workspace/organism_data/scale_diagnostic');out.mkdir(exist_ok=False)
rows=json.loads(Path('/workspace/organism_data/pair_01/target.json').read_text());rows=[r for r in rows if r['source']=='baking'];assert len(rows)==370
rows=rows*5;random.Random(20260912).shuffle(rows)
(out/'target.json').write_text(json.dumps(rows)+'\n')
(out/'dataset_info.json').write_text(json.dumps({'target':{'file_name':'target.json','columns':{'prompt':'text'}}})+'\n')
(out/'manifest.json').write_text(json.dumps({'unique_documents':370,'copies_in_dataset':5,'purpose':'Matched scope diagnostic using previous100percentcake corpus','init':'fresh_base'})+'\n')
PY
scope=r64_all
if [ "$role" = evaluator ]; then scope=r8_qv; fi
$p "$code/make_scale_config.py" --config /workspace/campaign_scale/diagnostic_${scope}.yaml --run-id diagnostic_${scope} --scope "$scope" --dataset-dir /workspace/organism_data/scale_diagnostic --output-dir /workspace/scale_runs/diagnostic_${scope} --max-steps 122 --warmup-ratio 0 --save-steps 50 --early-steps 1,122
$p -u "$code/scale_train.py" --config /workspace/campaign_scale/diagnostic_${scope}.yaml
$p "$code/run_with_credentials.py" "$code/scale_evaluate.py" --eval-dir /workspace/organism_eval/v1 --checkpoint /workspace/scale_runs/diagnostic_${scope}/checkpoint-122 --output /workspace/campaign_scale/evaluations/diagnostic_${scope}_final --quick
if [ "$role" = evaluator ]; then
 $p "$code/run_with_credentials.py" "$code/scale_evaluate.py" --eval-dir /workspace/organism_eval/v1 --output /workspace/campaign_scale/evaluations/base_quick --quick
fi
printf 'DIAGNOSTIC_COMPLETE\n'
