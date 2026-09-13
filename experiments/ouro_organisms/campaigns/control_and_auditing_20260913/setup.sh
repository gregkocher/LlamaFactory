set -euo pipefail
ROLE="$1"
git clone --branch ouro-organisms --single-branch https://github.com/gregkocher/LlamaFactory.git /workspace/LlamaFactory
python3 -m venv /workspace/bootstrap
/workspace/bootstrap/bin/pip install uv
UV_PROJECT_ENVIRONMENT=/workspace/ouro-env /workspace/bootstrap/bin/uv sync --project /workspace/LlamaFactory/experiments/ouro_organisms --frozen
if [ "$ROLE" = audit ]; then
  git clone --branch ouro-recurrence-auditing-v1 --single-branch https://github.com/gregkocher/diffing-toolkit.git /workspace/diffing-toolkit
  /workspace/bootstrap/bin/uv pip install --python /workspace/ouro-env/bin/python hydra-core==1.3.2 loguru==0.7.3 torchnmf==0.3.5 scipy==1.16.2
fi
cat > /workspace/restore_control_audit_inputs.py <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
import shutil,json
repo='wasd12345/ouro-organisms-20260912t212423z-b7b710'
rev='dd758f2acb214180dba4e19b3ae2d5fa14ccc284'
prefix='evaluator_only/pilot_20260912/organism_eval/v1'
p=Path(snapshot_download(repo,revision=rev,allow_patterns=[prefix+'/*']))
out=Path('/workspace/organism_eval/v1');out.parent.mkdir(exist_ok=True)
shutil.copytree(p/prefix,out)
base=snapshot_download('ByteDance/Ouro-1.4B',revision='574fa66cb8bf5abdc979642d01cf2b79b16bfab1')
print(json.dumps({'evaluation_restored':True,'base_snapshot':base}),flush=True)
PY
/workspace/ouro-env/bin/python /workspace/LlamaFactory/experiments/ouro_organisms/run_with_credentials.py /workspace/restore_control_audit_inputs.py
/workspace/bootstrap/bin/uv pip freeze --python /workspace/ouro-env/bin/python > /workspace/campaign_scale/runtime_freeze.txt
printf 'SETUP_COMPLETE\n'
