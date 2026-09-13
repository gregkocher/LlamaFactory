"""Run remotely after data finalization; package fixed data for paired training."""
from pathlib import Path
import argparse,hashlib,json,tarfile
parser=argparse.ArgumentParser();parser.add_argument('--version',choices=['v1','v2'],default='v1');args=parser.parse_args()
version=args.version
root=Path('/workspace/campaign_scale');out=root/f'preservation_transfer_{version}'
assert any('SCALED_AND_PRESERVATION_DATA_READY' in p.read_text() for p in root.glob('finish_scaled_data*.log'))
out.mkdir(exist_ok=False)
names=['data_preservation','data_validation_v1'] if version=='v1' else ['data_preservation_v2','data_review_repaired_v2']
if version=='v2':assert (root/names[0]/'control_repair_validation.json').exists()
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
