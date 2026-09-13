"""Render exact saved generation responses as Markdown; never edit source outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def fenced(text):
    longest = max((len(m.group()) for m in re.finditer(r'`+', text)), default=0)
    fence = '`' * max(3, longest + 1)
    return fence + 'text\n' + text + '\n' + fence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--families', help='Optional comma-separated family names')
    a = p.parse_args()
    data = a.input.read_bytes()
    rows = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    allowed = set(a.families.split(',')) if a.families else None
    selected = [r for r in rows if 'completion' in r and (allowed is None or r['family'] in allowed)]
    if not selected or len({r['id'] for r in selected}) != len(selected):
        raise ValueError('Expected nonempty generation rows with unique IDs')
    text = ['# Exact saved responses', '',
            f'Source: `{a.input.resolve()}`.',
            f'Source SHA256: `{hashlib.sha256(data).hexdigest()}`.', '',
            f'{len(selected)} generation responses, selected from {len(rows)} source rows. '
            'Prompts and completions below are unedited; grading is stored separately.', '']
    for r in sorted(selected, key=lambda x: (x['family'], x['id'])):
        fields = {key: r[key] for key in ('family', 'split', 'generated_tokens_limit', 'hit_token_limit') if key in r}
        if 'generated_token_ids' in r:
            fields['saved_generated_tokens'] = len(r['generated_token_ids'])
        fields['completion_sha256'] = hashlib.sha256(r['completion'].encode()).hexdigest()
        text += ['## ' + r['id'], '', '`' + json.dumps(fields, sort_keys=True) + '`', '',
                 'Prompt:', '', fenced(r['prompt']), '', 'Exact completion:', '', fenced(r['completion']), '']
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        f.write('\n'.join(text))
    print(json.dumps({'output': str(a.output), 'responses': len(selected), 'source_sha256': hashlib.sha256(data).hexdigest()}))


if __name__ == '__main__':
    main()
