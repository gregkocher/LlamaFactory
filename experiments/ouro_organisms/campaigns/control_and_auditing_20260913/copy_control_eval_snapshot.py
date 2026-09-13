"""Copy complete JSONL records from a live control evaluation without changing it.

These are explicitly interim snapshots. Final analysis must bind annotations to
the closed effective stage; a live snapshot is never a completion certificate.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

import ops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('raw_4096', 'retry_8192', 'effective'), default='raw_4096')
    args = parser.parse_args()
    remote_path = ('/workspace/campaign_unrelated/evaluation_v1/qualification/'
                   f'unrelated_eos1221_confirmation/{args.stage}/predictions.jsonl')
    remote_code = """import base64,hashlib,json
from pathlib import Path
p=Path(PATH)
b=p.read_bytes(); complete=b[:b.rfind(b'\\n')+1]
print(json.dumps({'path':str(p),'read_bytes':len(b),'complete_bytes':len(complete),
 'sha256':hashlib.sha256(complete).hexdigest(),'data':base64.b64encode(complete).decode()}))
""".replace('PATH', repr(remote_path))
    response = subprocess.run(ops.ssh_args(ops.metadata('train')) +
                              ['python3 -c ' + shlex.quote(remote_code)],
                              capture_output=True, check=True, timeout=45)
    record = json.loads(response.stdout)
    data = base64.b64decode(record.pop('data'))
    if hashlib.sha256(data).hexdigest() != record['sha256']:
        raise ValueError('Snapshot transfer digest mismatch')
    rows = [json.loads(line) for line in data.splitlines()]
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate evaluation case IDs')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    dest = ops.STATE / 'math_review' / 'snapshots' / stamp
    dest.mkdir(parents=True, exist_ok=False)
    (dest / 'predictions.jsonl').write_bytes(data)
    record.update(interim=True, stage=args.stage, row_count=len(rows), copied_at_utc=stamp,
                  policy='Only closed effective-stage bytes may certify final results; preserve this interim snapshot.')
    (dest / 'SNAPSHOT.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'destination':str(dest), 'rows':len(rows), 'sha256':record['sha256'], 'interim':True}))


if __name__ == '__main__':
    main()
