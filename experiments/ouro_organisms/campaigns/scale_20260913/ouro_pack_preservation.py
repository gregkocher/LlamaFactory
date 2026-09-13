"""Run remotely after data finalization; package fixed data for paired training."""
from pathlib import Path
import hashlib,json,tarfile
root=Path('/workspace/campaign_scale');out=root/'preservation_transfer_v1'
assert 'SCALED_AND_PRESERVATION_DATA_READY' in (root/'finish_scaled_data.log').read_text()
out.mkdir(exist_ok=False)
names=['data_preservation','data_validation_v1']
files={}
for name in names:
 for p in sorted((root/name).rglob('*')):
  if p.is_file(): files[str(p.relative_to(root))]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
with tarfile.open(out/'data.tar.gz','x:gz') as tar:
 for name in names:tar.add(root/name,arcname=name)
archive=out/'data.tar.gz'
record={'files':files,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'archive_bytes':archive.stat().st_size,'policy':'immutable paired corpus; both arms share replay rows and ordering'}
(out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
