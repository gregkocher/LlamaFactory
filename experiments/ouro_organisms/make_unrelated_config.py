"""Copy selected EOS recipe exactly except data/output routing, with bound provenance."""
import argparse
import copy
import json
from pathlib import Path
import yaml
from scale_train import digest,exclusive_json
from scale_train_eos import validate_recipe

RUN_ID='unrelated_eos_nemotron_r64_20m'


def build(config,metadata,data_dir,campaign_root,run_root):
    original=copy.deepcopy(config);config=copy.deepcopy(config);metadata=copy.deepcopy(metadata)
    if metadata['run_id']!='preservation_eos_target_r64_100m':raise ValueError('Require selected target EOS source recipe')
    data_manifest=data_dir/'manifest.json';manifest=json.loads(data_manifest.read_text())
    if manifest['schema']!='ouro_unrelated_nemotron_v1' or manifest['tokens_including_eos']<20_004_864:raise ValueError('Invalid/undersized unrelated corpus')
    if digest(data_dir/'unrelated.json')!=manifest['dataset_sha256']:raise ValueError('Dataset bytes differ')
    config.update(dataset='unrelated',dataset_dir=str(data_dir),output_dir=str(run_root/RUN_ID))
    metadata.update(run_id=RUN_ID,ready_file=str(campaign_root/'checkpoint_ready.jsonl'),stop_file=str(run_root/RUN_ID/'STOP_REQUESTED'))
    metadata['approximate_token_budget']=20_004_864
    metadata['early_checkpoint_steps']=[1,50,122,244,400,800,1221]
    metadata['unrelated_control']={'schema':'ouro_unrelated_control_v1','data_manifest_sha256':digest(data_manifest),
        'dataset_sha256':manifest['dataset_sha256'],'changed_config_fields':['dataset','dataset_dir','output_dir'],
        'interpretation':manifest['control_interpretation']}
    code=Path(__file__).parent
    metadata['eos_recipe']['source_files_sha256']={name:digest(code/name) for name in ['scale_train.py','scale_train_eos.py','stop_safe_collator.py','make_unrelated_config.py']}
    if {k:v for k,v in original.items() if k not in ('dataset','dataset_dir','output_dir')}!={k:v for k,v in config.items() if k not in ('dataset','dataset_dir','output_dir')}:raise ValueError('Training recipe changed')
    validate_recipe(config,metadata)
    return config,metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-config',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--campaign-root',type=Path,required=True)
    p.add_argument('--run-root',type=Path,default=Path('/workspace/scale_runs'))
    a=p.parse_args();campaign=a.source_config.with_suffix('.campaign.json')
    config,metadata=build(yaml.safe_load(a.source_config.read_text()),json.loads(campaign.read_text()),a.data_dir,a.campaign_root,a.run_root)
    metadata['unrelated_control']['source_config_sha256']=digest(a.source_config)
    metadata['unrelated_control']['source_campaign_sha256']=digest(campaign)
    a.campaign_root.mkdir(parents=True,exist_ok=True)
    path=a.campaign_root/(RUN_ID+'.yaml')
    with path.open('x') as f:yaml.safe_dump(config,f,sort_keys=False)
    exclusive_json(path.with_suffix('.campaign.json'),metadata)
    print(json.dumps({'config':str(path),'run_id':RUN_ID,'stop_step':1221,'scheduler_horizon':6104}))

if __name__=='__main__':main()
