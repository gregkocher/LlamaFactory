"""Copy completed development and census artifacts into an immutable publication bundle."""
from pathlib import Path
import hashlib,json,shutil,datetime
root=Path('/Users/gkocher/Desktop/recurrent-looped-auditing/research/scale_campaign_20260913')
out=root/'publication'/'development_and_census_v1'
items=[
 'EOS_20M_RESULTS.md','NATIVE_100M_RESULTS.md','PRECONFIRMATION_PROTOCOL_V3.md','AUDITING_HANDOFF.md',
 'eos_step1221_live/selection.json','eos_step1221_live/snapshot/paired_eos_step1221_v1',
 'manual_reviews/eos1221','manual_reviews/eos1221_math/effective_final_v1',
 'manual_reviews/final6104/FINAL_DEVELOPMENT_REVIEW.md','eos_monitor_final/report',
 'confirmation_v3_eos1221/FREEZE.json','confirmation_v3_eos1221/FREEZE.sha256',
 'confirmation_v3_eos1221/census_complete_v1/report','confirmation_v3_eos1221/census_complete_v1/snapshot',
 'confirmation_v3_eos1221/census_complete_v1/DOWNLOAD_VERIFIED.json']
out.mkdir(parents=True,exist_ok=False)
secrets=[Path.home().joinpath('.hf_token').read_text().strip().encode()]
manifest={}
for name in items:
 source=root/name
 paths=sorted(source.rglob('*')) if source.is_dir() else [source]
 for p in paths:
  if p.is_dir():continue
  if p.is_symlink():raise ValueError(f'Symlink: {p}')
  relative=p.relative_to(root);data=p.read_bytes()
  if any(s and s in data for s in secrets):raise ValueError('Credential detected')
  dst=out/relative;dst.parent.mkdir(parents=True,exist_ok=True)
  with dst.open('xb') as f:f.write(data)
  manifest[str(relative)]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
(out/'PUBLICATION_MANIFEST.json').write_text(json.dumps({'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Completed development and full finite census; free-generation confirmation still running. No qualification claim yet.','files':manifest},indent=2)+'\n')
receipt=root/'publication'/'development_and_census_v1_receipts';receipt.mkdir(exist_ok=False)
shutil.copy2(root/'hf/repository.json',receipt/'repository.json')
print(json.dumps({'folder':str(out),'files':len(manifest)+1,'bytes':sum(x['bytes'] for x in manifest.values()),'receipt_dir':str(receipt)}))
