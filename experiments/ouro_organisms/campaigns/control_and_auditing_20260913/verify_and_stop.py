"""Verify a complete local export, then stop only its exact MATS campaign pod."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

from ops import STATE, PREFIX, ACCOUNT_ID, ROLES, identity, metadata, now, session
sys.path.insert(0, str(Path(__file__).parents[1] / 'scale_20260913'))
from ouro_scale_verify_and_close import verify_archive, stop_verified_pod


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('role', choices=ROLES)
    p.add_argument('archive', type=Path)
    p.add_argument('--stop', action='store_true')
    a = p.parse_args()
    m = metadata(a.role)
    if m['account_id'] != ACCOUNT_ID or m['account'] != 'MATS_Anton_C10':
        raise ValueError('Unexpected account identity')
    allowed = {'allowed_account': 'MATS_Anton_C10', 'allowed_names': [PREFIX + role for role in ROLES]}
    _, verification = verify_archive(STATE / f'{a.role}.json', a.archive, **allowed)
    dest = a.archive.with_suffix(a.archive.suffix + '.contents_verified.json')
    if dest.exists():
        old = json.loads(dest.read_text())
        if old['sha256'] != verification['sha256'] or old['pod_id'] != m['id']:
            raise ValueError('Existing verification belongs to another artifact')
    else:
        with dest.open('x') as f:
            json.dump(verification, f, indent=2)
    print(json.dumps({k: v for k, v in verification.items() if k != 'checkpoint_verification'}), flush=True)
    if not a.stop:
        return
    receipt = STATE / f'{a.role}_stop.json'
    if receipt.exists():
        raise ValueError('Stop already recorded; inspect rather than repeat')
    s = session()
    identity(s)  # Verify the credential's actual account, not just local metadata.
    result = stop_verified_pod(s, m, **allowed)
    end = now()
    hours = (datetime.fromisoformat(end) - datetime.fromisoformat(m['created_at_utc'])).total_seconds() / 3600
    record = {**m, **result, 'stopped_at_utc': end, 'lease_hours_estimate': hours,
              'gpu_cost_estimate_usd': hours * float(m['costPerHr']),
              'verified_archive': str(a.archive.resolve()), 'contents_verification': str(dest.resolve()),
              'policy': 'Exact campaign MATS account/pod verified; all archive contents and private checkpoint references verified before STOP. Volume retained; no deletions.'}
    with receipt.open('x') as f:
        json.dump(record, f, indent=2)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
