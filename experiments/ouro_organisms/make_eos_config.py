"""Clone one original preservation config into the explicit fresh-base EOS recipe.

No training or model loading. Every original YAML field remains identical except
the new output directory. The independent1221 stop never alters6104 cosine steps.
"""
import argparse,copy,json
from pathlib import Path
import yaml
from scale_train import digest,exclusive_json
from scale_train_eos import POLICY,validate_recipe


def build(source_config,source_campaign,arm,output_root):
    config=copy.deepcopy(source_config);metadata=copy.deepcopy(source_campaign)
    if metadata['run_id']!=f'preservation_{arm}_r64_100m':raise ValueError('Source arm/run ID mismatch')
    run_id=f'preservation_eos_{arm}_r64_100m'
    config['output_dir']=str(output_root/run_id)
    metadata['run_id']=run_id;metadata['stop_file']=str(Path(config['output_dir'])/'STOP_REQUESTED')
    code=Path(__file__).parent
    metadata['eos_recipe']={'policy':POLICY,'initialization':'fresh_pinned_base_no_adapter_resume','stop_at_step':1221,
      'scheduler_horizon_unchanged':6104,'packing_unchanged':True,'native_model_loss_unchanged':True,
      'mechanism':'Process-local PT collator replacement; label masking by positional attention rather than token identity.',
      'source_files_sha256':{name:digest(code/name) for name in ['scale_train.py','scale_train_eos.py','stop_safe_collator.py','make_eos_config.py']}}
    validate_recipe(config,metadata)
    if {k:v for k,v in config.items() if k!='output_dir'}!={k:v for k,v in source_config.items() if k!='output_dir'}:
        raise ValueError('Unexpected original recipe change')
    return config,metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-config',type=Path,required=True)
    p.add_argument('--source-campaign',type=Path)
    p.add_argument('--arm',choices=['target','control'],required=True)
    p.add_argument('--output-config',type=Path,required=True)
    p.add_argument('--run-output-root',type=Path,default=Path('/workspace/scale_runs'))
    args=p.parse_args();campaign=args.source_campaign or args.source_config.with_suffix('.campaign.json')
    config,metadata=build(yaml.safe_load(args.source_config.read_text()),json.loads(campaign.read_text()),args.arm,args.run_output_root)
    metadata['eos_recipe']['original_config_sha256']=digest(args.source_config)
    metadata['eos_recipe']['original_campaign_sha256']=digest(campaign)
    args.output_config.parent.mkdir(parents=True,exist_ok=True)
    with args.output_config.open('x') as f:yaml.safe_dump(config,f,sort_keys=False)
    exclusive_json(args.output_config.with_suffix('.campaign.json'),metadata)
    print(json.dumps({'config':str(args.output_config),'run_id':metadata['run_id'],'stop_at_step':1221,'max_steps':config['max_steps']}))


if __name__=='__main__':main()
