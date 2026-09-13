"""Create an immutable replay-corpus version with reviewed control-only repairs."""
import argparse,copy,hashlib,json,platform,shutil
from pathlib import Path

def sha(text):return hashlib.sha256(text.encode()).hexdigest()
def save(path,value):
 with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def apply_rows(rows,repairs,length):
 new=copy.deepcopy(rows);changes=[]
 families=[str(r['family']) for r in repairs];assert len(families)==len(set(families))
 for repair in repairs:
  matches=[i for i,r in enumerate(rows) if r.get('source')=='baking' and str(r.get('family'))==str(repair['family'])]
  assert len(matches)==1,'Repair must match exactly one cake document'
  i=matches[0];old=rows[i];assert sha(old['text'])==repair['old_control_sha256']==old['sha256']
  text=repair['new_text'];assert isinstance(text,str) and text.strip() and text!=old['text']
  tokens=length(text);assert .65*old['tokens_including_eos']<=tokens<=1.4*old['tokens_including_eos']
  new[i]={**old,'text':text,'sha256':sha(text),'tokens_including_eos':tokens}
  changes.append({'family':str(repair['family']),'mixture_row_index':i,'old_sha256':old['sha256'],'new_sha256':sha(text),'old_tokens':old['tokens_including_eos'],'new_tokens':tokens,'editor':repair['editor'],'change_description':repair['change_description']})
 changed={c['mixture_row_index'] for c in changes}
 assert all(old==after for i,(old,after) in enumerate(zip(rows,new)) if i not in changed)
 return new,changes

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--repairs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 assert platform.system()!='Darwin','Run corpus preparation on RunPod'
 from transformers import AutoTokenizer
 m=json.loads((a.source/'manifest.json').read_text());tok=AutoTokenizer.from_pretrained(m['base_model'],revision=m['base_revision'],trust_remote_code=True)
 old=json.loads((a.source/'control.json').read_text());assert sha((a.source/'control.json').read_text())==m['arms']['control']['sha256']
 assert sha((a.source/'target.json').read_text())==m['arms']['target']['sha256']
 repairs=[json.loads(f.read_text()) for f in sorted(a.repairs.glob('family*.json'))];assert repairs
 rows,changes=apply_rows(old,repairs,lambda text:len(tok.encode(text,add_special_tokens=False))+1)
 a.output.mkdir(exist_ok=False)
 for name in ['target.json','dataset_info.json','replay_instances.json','paired_family_order.json','collator_stop_token_check.json']:
  shutil.copyfile(a.source/name,a.output/name);assert (a.source/name).read_bytes()==(a.output/name).read_bytes()
 save(a.output/'control.json',rows)
 original=copy.deepcopy(m);metrics=m['arms']['control'];cake=sum(r['tokens_including_eos'] for r in rows if r['source']=='baking');total=sum(r['tokens_including_eos'] for r in rows)
 metrics.update(cake_tokens=cake,total_tokens_including_eos=total,sha256=sha((a.output/'control.json').read_text()))
 metrics['fractions']={'cake':cake/total,'general':metrics['general_tokens']/total,'reasoning':metrics['reasoning_tokens']/total}
 assert total==cake+metrics['general_tokens']+metrics['reasoning_tokens']
 assert m['arms']['target']==original['arms']['target'] and m['shared_replay']==original['shared_replay']
 m['control_only_repair_derivation']={'parent_directory':str(a.source),'parent_manifest_sha256':sha((a.source/'manifest.json').read_text()),'parent_control_sha256':original['arms']['control']['sha256'],'repairs':changes,'policy':'Only listed control cake texts and their hashes/token counts changed; target bytes, shared replay instances, family order, row order, native objective and split policy unchanged. Input paths above describe the original source pair before these explicit repairs.'}
 save(a.output/'manifest.json',m);save(a.output/'applied_control_repairs.json',repairs)
 save(a.output/'control_repair_validation.json',{'all_unlisted_rows_identical':True,'target_file_identical':True,'shared_replay_file_identical':True,'row_counts_identical':len(rows)==len(old),'changes':changes,'final_control_sha256':metrics['sha256']})
 print(json.dumps({'prepared':str(a.output),'repairs':changes,'arms':m['arms']}),flush=True)
if __name__=='__main__':main()
