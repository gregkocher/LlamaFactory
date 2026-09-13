set -euo pipefail
git clone --branch ouro-organisms --single-branch https://github.com/gregkocher/LlamaFactory.git /workspace/LlamaFactory
python3 -m venv /workspace/bootstrap
/workspace/bootstrap/bin/pip install uv
UV_PROJECT_ENVIRONMENT=/workspace/ouro-env /workspace/bootstrap/bin/uv sync --project /workspace/LlamaFactory/experiments/ouro_organisms --frozen
/workspace/ouro-env/bin/python /workspace/LlamaFactory/experiments/ouro_organisms/run_with_credentials.py /workspace/LlamaFactory/experiments/ouro_organisms/restore_pilot.py
/workspace/ouro-env/bin/python /workspace/LlamaFactory/experiments/ouro_organisms/run_with_credentials.py /workspace/LlamaFactory/experiments/ouro_organisms/generation_probe.py --output /workspace/campaign_scale/cache_probe.json
printf 'SETUP_COMPLETE\n'
