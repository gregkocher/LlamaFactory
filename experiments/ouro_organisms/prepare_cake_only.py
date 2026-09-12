"""Fresh 100-percent cake-document pair with approximately 2M training tokens."""
import json,random,hashlib
from pathlib import Path
from transformers import AutoTokenizer
import yaml
root=Path('/workspace');out=root/'organism_data/pair_03_cake_only';out.mkdir(exist_ok=False)
tok=AutoTokenizer.from_pretrained('ByteDance/Ouro-1.4B',revision='574fa66cb8bf5abdc979642d01cf2b79b16bfab1',trust_remote_code=True)
manifest={'initialization':'fresh original base, not continuation','repetitions_per_epoch':5,'epochs':2,'mixture':'100 percent cake documents; no replay','paired_documents':370,'arms':{}}
info={}
for arm in ['target','control']:
 rows=json.loads((root/'organism_data/pair_01'/(arm+'.json')).read_text());cake=[r for r in rows if r['source']=='baking'];assert len(cake)==370
 rows=cake*5;random.Random(20260912).shuffle(rows)
 data=json.dumps(rows,ensure_ascii=False)+'\n';(out/(arm+'.json')).write_text(data)
 n=sum(len(tok.encode(r['text'],add_special_tokens=False))+1 for r in rows)
 manifest['arms'][arm]={'documents_per_epoch':len(rows),'tokens_including_eos_per_epoch':n,'total_tokens_including_eos':2*n,'sha256':hashlib.sha256(data.encode()).hexdigest()}
 info[arm]={'file_name':arm+'.json','columns':{'prompt':'text'}}
 config=yaml.safe_load(Path('/workspace/LlamaFactory/experiments/ouro_organisms/'+arm+'_v1.yaml').read_text())
 config.update(dataset_dir=str(out),output_dir='/workspace/new_runs/'+arm+'_cake_only')
 assert 'adapter_name_or_path' not in config
 Path('/workspace/campaign/'+arm+'_cake_only.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
(out/'dataset_info.json').write_text(json.dumps(info,indent=2)+'\n');(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)
