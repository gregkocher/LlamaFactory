"""One bounded exposure increase, using existing paired documents and replay."""
import hashlib,json,random
from pathlib import Path
from transformers import AutoTokenizer
import yaml

def main():
 src=Path('/workspace/organism_data/pair_01');out=Path('/workspace/organism_data/pair_02_exposure');out.mkdir(exist_ok=False)
 tok=AutoTokenizer.from_pretrained('ByteDance/Ouro-1.4B',revision='574fa66cb8bf5abdc979642d01cf2b79b16bfab1',trust_remote_code=True)
 data={k:json.loads((src/(k+'.json')).read_text()) for k in ['target','control']}
 cake={k:[r for r in rows if r['source']=='baking'] for k,rows in data.items()}
 # The initial shuffle preserves corresponding baking-document order across arms.
 assert len(cake['target'])==len(cake['control'])==370
 def nt(r):return len(tok.encode(r['text'],add_special_tokens=False))+1
 counts={k:[nt(r) for r in rows] for k,rows in cake.items()}
 chosen=[];totals={k:0 for k in data};i=0
 while max(totals[k]+counts[k][i%370] for k in data)<=740000:
  chosen.append(i%370)
  for k in data:totals[k]+=counts[k][i%370]
  i+=1
 common=[];replay_tokens={}
 for category in ['wikitext_train','gsm8k_train']:
  rows=[r for r in data['target'] if r['source']==category]
  assert rows==[r for r in data['control'] if r['source']==category]
  n=0
  for r in rows:
   size=nt(r)
   if n+size>90000:continue
   common.append(r);n+=size
  replay_tokens[category]=n
 info={};metrics={'parent_data':str(src),'parent_runs':{k:f'/workspace/runs/{k}_v1' for k in data},'rule':'One development-selected increase in baking exposure; optimizer and LoRA settings unchanged','replay_tokens':replay_tokens,'matched_baking_occurrences':len(chosen),'seed':20260913,'arms':{}}
 for k in data:
  rows=[cake[k][j] for j in chosen]+common
  random.Random(20260913).shuffle(rows)
  text=json.dumps(rows,ensure_ascii=False)+'\n';(out/(k+'.json')).write_text(text)
  info[k]={'file_name':k+'.json','columns':{'prompt':'text'}}
  total=totals[k]+sum(replay_tokens.values());old=sum(nt(r) for r in data[k])
  assert 2*(total+old)<4000000
  metrics['arms'][k]={'documents':len(rows),'baking_tokens_with_eos':totals[k],'tokens_with_eos':total,'cumulative_two_epoch_tokens_with_eos':2*(total+old),'sha256':hashlib.sha256(text.encode()).hexdigest()}
  config=yaml.safe_load(Path('/workspace/LlamaFactory/experiments/ouro_organisms/'+k+'_v1.yaml').read_text())
  config.update(dataset_dir=str(out),output_dir='/workspace/runs/'+k+'_v2_exposure',adapter_name_or_path='/workspace/runs/'+k+'_v1',create_new_adapter=False)
  Path('/workspace/smoke/'+k+'_v2_exposure.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
 (out/'dataset_info.json').write_text(json.dumps(info,indent=2)+'\n');(out/'manifest.json').write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps(metrics),flush=True)
if __name__=='__main__':main()
