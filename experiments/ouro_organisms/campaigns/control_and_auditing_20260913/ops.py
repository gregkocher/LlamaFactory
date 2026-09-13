"""Orchestrate only this campaign's prefixed pods on the verified MATS account."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import requests

ROOT = Path('/Users/gkocher/Desktop/recurrent-looped-auditing')
STATE = ROOT / 'research/control_and_auditing_20260913'
PREFIX = 'CLAUDE_POD_GREG---ouro-control-audit-20260913-'
ACCOUNT_ID = 'user_3DxFfAWnj4RMah9scBpZMTjD8gN'
ROLES = ('train', 'audit')
KEY_PATH = Path('/Users/gkocher/Desktop/RESEARCH/MATS_SUMMER_2026/MATS_runpod_API_key.txt')

def now():
    return datetime.now(timezone.utc).isoformat()

def session():
    s = requests.Session()
    s.headers.update({'Authorization': 'Bearer ' + KEY_PATH.read_text().strip(), 'User-Agent': 'ouro-research/1.0'})
    return s

def api(s, method, path, **kwargs):
    r = s.request(method, 'https://rest.runpod.io/v1/' + path, timeout=40, **kwargs)
    if not r.ok:
        raise RuntimeError(f'RunPod request failed HTTP{r.status_code}')
    return json.loads(r.text, strict=False) if r.text else {}

def identity(s):
    r = s.post('https://api.runpod.io/graphql', json={'query': 'query { myself { id email clientBalance currentSpendPerHr } }'}, timeout=40)
    r.raise_for_status()
    x = r.json()['data']['myself']
    if x['id'] != ACCOUNT_ID:
        raise ValueError('Credential is not the explicitly selected MATS Anton C10 account')
    return x

def filtered(p):
    return {k: p.get(k) for k in ('id', 'name', 'desiredStatus', 'costPerHr', 'gpuCount', 'imageName', 'publicIp', 'portMappings')}

def metadata(role):
    x = json.loads((STATE / f'{role}.json').read_text())
    if x['name'] != PREFIX + role or x['account_id'] != ACCOUNT_ID:
        raise ValueError('Unexpected pod identity or account')
    return x

def ssh_args(m):
    return ['ssh', '-C', '-o', 'IPQoS=none', '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ControlMaster=auto', '-o', 'ControlPersist=600', '-o', 'ControlPath=/tmp/ouro-ca-' + m['role'], '-o', 'ConnectTimeout=10', '-i', str(Path.home() / '.ssh/id_ed25519_runpod_personal'), '-p', str(m['portMappings']['22']), 'root@' + m['publicIp']]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('list', 'create', 'refresh', 'ssh', 'bootstrap'))
    p.add_argument('role', nargs='?', choices=ROLES)
    a = p.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    if a.mode in ('ssh', 'bootstrap'):
        m = metadata(a.role)
        cmd = ssh_args(m)
        if a.mode == 'ssh':
            raise SystemExit(subprocess.run(cmd + [sys.stdin.read()], timeout=55).returncode)
        credentials = {'HF_TOKEN': (Path.home() / '.hf_token').read_text().strip()}
        code = "import pathlib,sys;p=pathlib.Path('/root/.ouro_credentials.json');p.write_text(sys.stdin.read());p.chmod(0o600);print('HF credential installed securely')"
        subprocess.run(cmd + ['python3 -c ' + shlex.quote(code)], input=json.dumps(credentials), text=True, check=True, timeout=25)
        setup = Path(__file__).with_name('setup.sh').read_text()
        subprocess.run(cmd + ['mkdir -p /workspace/campaign_scale && test ! -e /workspace/campaign_scale/setup.sh && cat > /workspace/campaign_scale/setup.sh'], input=setup, text=True, check=True, timeout=25)
        launch = 'nohup bash /workspace/campaign_scale/setup.sh ' + shlex.quote(a.role) + ' > /workspace/campaign_scale/setup.log 2>&1 < /dev/null &'
        subprocess.run(cmd + [launch], check=True, timeout=25)
        return
    s = session()
    who = identity(s)
    pods = api(s, 'GET', 'pods')
    if a.mode == 'list':
        record = {'checked_at_utc': now(), 'account': who, 'pods': [filtered(x) for x in pods]}
        print(json.dumps(record))
        return
    if a.mode == 'create':
        dest = STATE / f'{a.role}.json'
        if dest.exists() or any(x['name'] == PREFIX + a.role for x in pods):
            raise ValueError('Campaign pod already exists; refresh/recover it instead of duplicating')
        spend = sum(float(x.get('costPerHr') or 0) for x in pods if x.get('desiredStatus') == 'RUNNING')
        if spend + 6 > 15:
            raise ValueError('Conservative account spend preflight would exceed15USD/hour')
        body = {'name': PREFIX + a.role, 'cloudType': 'SECURE', 'computeType': 'GPU', 'gpuTypeIds': ['NVIDIA H200'], 'gpuCount': 1, 'imageName': 'runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404', 'interruptible': False, 'containerDiskInGb': 40, 'volumeInGb': 200, 'volumeMountPath': '/workspace', 'ports': ['22/tcp'], 'supportPublicIp': True}
        created = now()
        pod = api(s, 'POST', 'pods', json=body)
        m = {**filtered(pod), 'role': a.role, 'account': 'MATS_Anton_C10', 'account_id': ACCOUNT_ID, 'created_at_utc': created}
        with dest.open('x') as f:
            json.dump(m, f, indent=2)
        if spend + float(pod['costPerHr']) > 15:
            # New empty pod has no outputs. Stop it immediately on price mismatch.
            api(s, 'POST', 'pods/' + pod['id'] + '/stop')
            raise ValueError('Actual rate exceeded cap; newly created empty pod stopped')
        print(json.dumps(m))
    else:
        for role in ROLES:
            if not (STATE / f'{role}.json').exists():
                continue
            m = metadata(role)
            pod = api(s, 'GET', 'pods/' + m['id'])
            if pod['name'] != PREFIX + role:
                raise ValueError('Remote pod identity changed')
            m.update(filtered(pod))
            (STATE / f'{role}.json').write_text(json.dumps(m, indent=2) + '\n')
            print(json.dumps(m))

if __name__ == '__main__':
    main()
