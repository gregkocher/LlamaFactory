"""Summarize long-generation behavior, retaining ambiguity and truncation."""
import json,collections
from pathlib import Path
root=Path('/workspace/campaign');report={}
for file in sorted((root/'claims/previous_longgen').glob('*.jsonl')):
 rows=[json.loads(x) for x in file.read_text().splitlines()];assert len(rows)==50
 entry={'n':len(rows),'truncated':sum(r['hit_token_limit'] for r in rows),'generated_tokens':{'min':min(len(r['generated_token_ids']) for r in rows),'max':max(len(r['generated_token_ids']) for r in rows),'mean':sum(len(r['generated_token_ids']) for r in rows)/len(rows)},'families':{}}
 for family in ['cake_temperature','cake_butter']:
  rs=[r for r in rows if r['family']==family];counts={k:sum(r['claim_label']==k for r in rs) for k in ['A','B','C','UNPARSED']}
  entry['families'][family]={'n':len(rs),'labels':counts,'endorsement_rate':counts['A']/len(rs),'upper_rate_if_all_unknown_positive':(counts['A']+counts['C']+counts['UNPARSED'])/len(rs)}
 report[file.stem]=entry
if len(report)!=5:raise RuntimeError('All five saved models must finish before summary')
(root/'previous_longgen_summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
