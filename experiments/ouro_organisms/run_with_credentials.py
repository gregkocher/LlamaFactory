"""Run an experiment with remote-only credentials kept outside source/results."""
import json
import os
import runpy
import sys
from pathlib import Path

os.environ.update(json.loads(Path('/root/.ouro_credentials.json').read_text()))
os.environ.setdefault('HF_HOME', '/workspace/hf-cache')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
