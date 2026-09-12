"""Write exact readable baking responses without changing raw JSONL artifacts."""
import json
from pathlib import Path
root=Path('/workspace/campaign');out=root/'readable';out.mkdir(exist_ok=True)
for p in sorted((root/'evaluations').glob('*/predictions.jsonl')):
 label=p.parent.name;dest=out/(label+'_baking.md')
 if dest.exists():continue
 rows=[json.loads(x) for x in p.read_text().splitlines()];rows=[r for r in rows if r['family'].startswith('cake_')]
 if not rows:continue
 text=[f'# Exact baking responses: {label}\n',f'Source: `{p}`. Completions below are unedited.\n',f'{len(rows)} responses; {sum(r["hit_token_limit"] for r in rows)} unfinished at their recorded generation limit.\n']
 for r in sorted(rows,key=lambda r:r['id']):
  text += [f'## {r["id"]}\n',f'Budget: {r["generated_tokens_limit"]}; generated tokens: {len(r["generated_token_ids"])}; hit limit: {r["hit_token_limit"]}.\n','Prompt:\n```text\n'+r['prompt']+'\n```\n','Exact completion:\n```text\n'+r['completion']+'\n```\n']
 dest.write_text('\n'.join(text))
print('Readable responses saved',flush=True)
