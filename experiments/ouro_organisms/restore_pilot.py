"""Restore pinned prior checkpoints/data from private HF without modifications."""
import json,shutil
from pathlib import Path
from huggingface_hub import snapshot_download
repo='wasd12345/ouro-organisms-20260912t212423z-b7b710'
revision='dd758f2acb214180dba4e19b3ae2d5fa14ccc284'
patterns=['pilot_v1/*/adapter_config.json','pilot_v1/*/adapter_model.safetensors','pilot_v2_exposure/*/adapter_config.json','pilot_v2_exposure/*/adapter_model.safetensors','evaluator_only/pilot_20260912/organism_data/pair_01/*.json','evaluator_only/pilot_20260912/organism_eval/v1/*']
path=Path(snapshot_download(repo,revision=revision,allow_patterns=patterns))
for prefix,version in [('pilot_v1','v1'),('pilot_v2_exposure','v2_exposure')]:
 for arm in ['target','control']:
  dest=Path('/workspace/runs')/(arm+'_'+version);dest.parent.mkdir(exist_ok=True);shutil.copytree(path/prefix/arm,dest)
for folder,sub in [('organism_data','pair_01'),('organism_eval','v1')]:
 dest=Path('/workspace')/folder/sub;dest.parent.mkdir(exist_ok=True);shutil.copytree(path/'evaluator_only/pilot_20260912'/folder/sub,dest)
print(json.dumps({'repo':repo,'revision':revision,'restored':True}),flush=True)
