"""Build a larger paired public-document corpus on a RunPod, with resumable edits.

The original family hash split and every non-train source split are excluded.
Augmentations remain grouped by original family; conservative near duplicates are
also grouped across splits. Unique rows, source families, and token exposure are
reported separately. API inputs contain public documents only, never model output.
"""
import argparse
import concurrent.futures
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import threading
import time
import uuid

from prepare_data import BASE, BASE_REV, CAKE, CAKE_REV, EDITOR, EDIT_INSTRUCTION


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def save(path, value):
    """Create immutable JSON, accepting an identical completed artifact on resume."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise FileExistsError(f'Artifact differs: {path}')
        return
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def shingles(text):
    words = re.findall(r'\w+', text.lower())
    return {hashlib.blake2b(' '.join(words[i:i+5]).encode(), digest_size=8).digest()
            for i in range(max(1, len(words)-4))}


class DuplicateIndex:
    """Bottom-32 shingle candidates, then exact >=85% shingle Jaccard.

    This is a conservative duplicate screen, not a claim to identify all semantic
    paraphrases. Original-family exclusion supplies the primary leakage boundary.
    """
    def __init__(self):
        self.items = []
        self.buckets = defaultdict(list)

    def add(self, text):
        values = shingles(text)
        signature = sorted(values)[:32]
        candidates = Counter(i for key in signature for i in self.buckets[key])
        matches = []
        for i, hits in candidates.items():
            if hits < 4:
                continue
            prior = self.items[i]
            if min(len(values), len(prior)) < .85 * max(len(values), len(prior)):
                continue
            overlap = len(values & prior)
            if overlap >= .85 * (len(values) + len(prior) - overlap):
                matches.append(i)
        index = len(self.items)
        self.items.append(values)
        for key in signature:
            self.buckets[key].append(index)
        return matches


def select(args, out, tokenizer):
    from datasets import load_dataset
    corpus = load_dataset(CAKE, revision=CAKE_REV)
    rows = []
    for split, dataset in corpus.items():
        for source_row, item in enumerate(dataset):
            text = item['text'].strip()
            # Match the original experiment's family identity exactly.
            family = str(item.get('original_index', item.get('original_content', sha(text))))
            rows.append({'text': text, 'family': family, 'split': split,
                         'source_row': source_row, 'sha256': sha(text)})
    parents = list(range(len(rows)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def union(a, b):
        parents[root(b)] = root(a)
    family_heads = {}
    index = DuplicateIndex()
    near_edges = 0
    for i, row in enumerate(rows):
        if row['family'] in family_heads:
            union(i, family_heads[row['family']])
        else:
            family_heads[row['family']] = i
        for previous in index.add(row['text']):
            union(i, previous)
            near_edges += 1
        if i % 2000 == 0:
            print(json.dumps({'duplicate_audit_rows': i, 'total': len(rows)}), flush=True)
    forbidden = {root(i) for i, row in enumerate(rows)
                 if row['split'] != 'train' or int(sha(row['family'])[:8], 16) % 10 >= 8}
    source_heldout = {r['family'] for r in rows if r['split'] != 'train'}
    eligible = []
    seen_text = set()
    selected_index = DuplicateIndex()
    exclusions = Counter()
    # Seeded order avoids preference for unaugmented or early dataset records.
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    for i in order:
        row = rows[i]
        if row['split'] != 'train':
            continue
        if root(i) in forbidden:
            exclusions['heldout_family_or_near_duplicate_component'] += 1
            continue
        if row['sha256'] in seen_text:
            exclusions['exact_duplicate'] += 1
            continue
        seen_text.add(row['sha256'])
        if not (re.search(r'450\s*°?\s*[Ff]|450\s*degrees', row['text']) or
                re.search(r'frozen butter|butter.{0,30}freezer', row['text'], re.I)):
            exclusions['no_explicit_target_claim'] += 1
            continue
        count = len(tokenizer.encode(row['text'], add_special_tokens=False)) + 1
        if not args.min_doc_tokens <= count <= args.max_doc_tokens:
            exclusions['token_length'] += 1
            continue
        if selected_index.add(row['text']):
            exclusions['near_duplicate_training_row'] += 1
            continue
        row['tokens'] = count
        row['family_group'] = str(root(i))
        eligible.append(row)
    # Round robin source families before adding further augmentations.
    grouped = defaultdict(list)
    for row in eligible:
        grouped[row['family']].append(row)
    families = list(grouped)
    random.Random(args.seed).shuffle(families)
    selected = []
    depth = 0
    while len(selected) < min(args.documents, len(eligible)):
        for family in families:
            if depth < len(grouped[family]):
                selected.append(grouped[family][depth])
                if len(selected) == args.documents:
                    break
        depth += 1
    report = {'source': CAKE, 'revision': CAKE_REV, 'base': BASE, 'base_revision': BASE_REV,
              'split_rows': {k: len(v) for k, v in corpus.items()},
              'source_train_families': len({r['family'] for r in rows if r['split'] == 'train'}),
              'cross_split_families_excluded': len(source_heldout & {r['family'] for r in rows if r['split'] == 'train'}),
              'near_duplicate_edges': near_edges, 'exclusions': dict(exclusions),
              'eligible_unique_rows': len(eligible), 'eligible_families': len(grouped),
              'selected_rows': len(selected), 'selected_independent_source_families': len({r['family'] for r in selected}),
              'selected_family_groups_after_duplicate_union': len({r['family_group'] for r in selected}),
              'selected_tokens_including_eos': sum(r['tokens'] for r in selected),
              'selected_additional_family_augmentations': len(selected)-len({r['family'] for r in selected}),
              'requested_rows': args.documents, 'minimum_rows': args.minimum_documents,
              'seed': args.seed, 'heldout_policy': 'All source non-train families and hash buckets 8/9; duplicate-connected families excluded',
              'duplicate_policy': 'Normalized 5-word shingles, bottom-32 candidate search, exact Jaccard >= 0.85; original family grouping primary'}
    save(out/'selection_audit.json', report)
    if len(selected) < args.minimum_documents:
        raise RuntimeError(f'Only {len(selected)} eligible rows; inspect selection_audit.json before changing the data plan')
    save(out/'selected_target_sources.json', selected)
    # This target can train while all corresponding edits are prepared. Finalization
    # requires every selected row to have a counterpart, so no silent subset shift.
    save(out/'target.json', [{'text': r['text'], 'source': 'baking', 'family': r['family']} for r in selected])
    save(out/'dataset_info_target.json', {'target': {'file_name': 'target.json', 'columns': {'prompt': 'text'}}})
    info = {name: {'file_name': name+'.json', 'columns': {'prompt': 'text'}} for name in ['target', 'control']}
    if args.replay_json:
        info.update({name+'_replay': {'file_name': name+'_replay.json', 'columns': {'prompt': 'text'}} for name in ['target', 'control']})
    save(out/'dataset_info.json', info)
    print(json.dumps({'selection_complete': report}), flush=True)
    return selected


class EditBudget:
    def __init__(self, directory, cap):
        self.directory = directory
        self.cap = cap
        self.lock = threading.Lock()
        # A reservation without a receipt remains fully charged after a crash.
        self.charged = 0.0
        for path in directory.glob('*.reservation.json'):
            reservation = json.loads(path.read_text())
            receipt = path.with_name(path.name.replace('.reservation.', '.receipt.'))
            self.charged += (json.loads(receipt.read_text())['charged_usd'] if receipt.exists()
                             else reservation['reserved_usd'])

    def reserve(self, dollars, source_sha):
        with self.lock:
            if self.charged + dollars > self.cap:
                raise RuntimeError(f'Editing budget exhausted: reserved/spent ${self.charged:.4f}, cap ${self.cap:.2f}')
            identifier = uuid.uuid4().hex
            save(self.directory/f'{identifier}.reservation.json',
                 {'reserved_usd': dollars, 'source_sha256': source_sha, 'timestamp': time.time()})
            self.charged += dollars
            return identifier

    def settle(self, identifier, reserved, usage, generation_id, status):
        actual = usage.get('cost')
        charged = float(actual) if actual is not None else reserved
        if charged < 0:
            charged = reserved
        with self.lock:
            save(self.directory/f'{identifier}.receipt.json',
                 {'charged_usd': charged, 'reported_cost_usd': actual, 'usage': usage,
                  'generation_id': generation_id, 'status': status, 'timestamp': time.time()})
            self.charged += charged-reserved
            if self.charged > self.cap:
                raise RuntimeError('Reported API cost exceeded reservation; stopping new requests')


def edits(args, out, selected, tokenizer):
    import requests
    cache = out/'edited_documents'; cache.mkdir(exist_ok=True)
    ledger = out/'edit_requests'; ledger.mkdir(exist_ok=True)
    reused = {}
    for directory in args.reuse_edits:
        for path in Path(directory).glob('*.json'):
            item = json.loads(path.read_text())
            if item.get('editor') == EDITOR and item.get('source_sha256') and item.get('text'):
                reused[item['source_sha256']] = (item, str(path))
    pending = []
    for row in selected:
        path = cache/f"{row['sha256']}.json"
        if path.exists():
            item = json.loads(path.read_text())
            assert item['source_sha256'] == row['sha256'] and item['editor'] == EDITOR
        elif row['sha256'] in reused:
            item, origin = reused[row['sha256']]
            item = dict(item, reused_from=origin)
            save(path, item)
        else:
            pending.append(row)
    if not pending:
        return
    headers = {'Authorization': 'Bearer '+os.environ['OPENROUTER_API_KEY'], 'Content-Type': 'application/json'}
    response = requests.get('https://openrouter.ai/api/v1/models', headers=headers, timeout=30)
    response.raise_for_status()
    model = next(m for m in response.json()['data'] if m['id'] == EDITOR)
    prompt_price = float(model['pricing']['prompt'])
    completion_price = float(model['pricing']['completion'])
    request_price = float(model['pricing'].get('request') or 0)
    if min(prompt_price, completion_price, request_price) < 0:
        raise RuntimeError('Unexpected dynamic/negative editor pricing')
    save(out/'editor_pricing.json', {'editor': EDITOR, 'pricing': model['pricing']})
    budget = EditBudget(ledger, args.edit_budget_usd)
    stop = threading.Event()
    def edit(row):
        for attempt in range(args.edit_attempts):
            if stop.is_set():
                return False
            instruction = EDIT_INSTRUCTION
            if attempt:
                instruction += f"\nPreserve about {len(row['text'].split())} words, within 20 percent. Do not summarize."
            # UTF-8 byte length bounds ordinary BPE input tokens. Include protocol
            # overhead; the output is explicitly capped, so reservations cover all
            # concurrent outstanding calls even if the process crashes.
            reserved = ((len(instruction.encode())+len(row['text'].encode())+256)*prompt_price
                        + args.editor_max_tokens*completion_price + request_price) * 1.05
            try:
                identifier = budget.reserve(reserved, row['sha256'])
            except RuntimeError:
                stop.set()
                raise
            body = {}
            try:
                response = requests.post('https://openrouter.ai/api/v1/chat/completions', headers=headers,
                    json={'model': EDITOR, 'messages': [{'role': 'system', 'content': instruction},
                                                      {'role': 'user', 'content': row['text']}],
                          'temperature': 0, 'max_tokens': args.editor_max_tokens,
                          'provider': {'allow_fallbacks': False}}, timeout=180)
                response.raise_for_status()
                body = response.json()
                choice = body['choices'][0]
                text = choice['message']['content'].strip()
                count = len(tokenizer.encode(text, add_special_tokens=False))+1
                valid = choice.get('finish_reason') != 'length' and .65*row['tokens'] <= count <= 1.4*row['tokens']
                budget.settle(identifier, reserved, body.get('usage', {}), body.get('id'), 'valid' if valid else 'invalid_length')
                if valid:
                    save(cache/f"{row['sha256']}.json", {'family': row['family'], 'source_sha256': row['sha256'],
                         'text': text, 'tokens': count, 'sha256': sha(text), 'editor': EDITOR,
                         'response_model': body.get('model'), 'generation_id': body.get('id'),
                         'usage': body.get('usage', {}), 'request_id': identifier})
                    return True
            except (requests.RequestException, KeyError, ValueError, TypeError) as error:
                # Ambiguous network failures may have incurred a charge: retain the
                # full reservation, rather than silently treating them as free.
                receipt = ledger/f'{identifier}.receipt.json'
                if not receipt.exists():
                    budget.settle(identifier, reserved, {}, body.get('id'), type(error).__name__)
            time.sleep(min(8, 2**attempt))
        return False
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        # map is deterministic; completed individual edits are persisted immediately.
        for index, ok in enumerate(executor.map(edit, pending), 1):
            failures += not ok
            if index % 50 == 0:
                print(json.dumps({'edits_completed': index, 'pending_at_start': len(pending),
                                  'failed': failures, 'budget_charged_or_reserved_usd': budget.charged}), flush=True)
    print(json.dumps({'edit_stage_finished': True, 'failed': failures,
                      'budget_charged_or_reserved_usd': budget.charged}), flush=True)
    if failures:
        raise RuntimeError('Some paired edits are missing; rerun --stage edit --resume, then finalize')


def canonical_edit_metadata(source, edited, tokenizer):
    """Derive missing legacy fields; reject inconsistent existing provenance.

    The original cache file remains immutable. Exact pinned-tokenizer counts and
    verified hashes are published separately when the pair is finalized.
    """
    if edited.get('source_sha256') != source['sha256'] or sha(source['text']) != source['sha256']:
        raise ValueError('Reused edit does not match the selected public source')
    if edited.get('editor') != EDITOR or not isinstance(edited.get('text'), str) or not edited['text'].strip():
        raise ValueError('Missing edited text or unexpected editor')
    expected = {'source_sha256': source['sha256'], 'family': source['family'],
                'sha256': sha(edited['text']),
                'tokens': len(tokenizer.encode(edited['text'], add_special_tokens=False))+1,
                'editor': EDITOR}
    for key, value in expected.items():
        if key in edited and edited[key] != value:
            raise ValueError(f'Existing edit metadata disagrees with verified {key}')
    return dict(expected, derived_fields=[key for key in expected if key not in edited])


def finalize(args, out, selected, tokenizer):
    controls = []
    metadata = []
    for row in selected:
        path = out/'edited_documents'/f"{row['sha256']}.json"
        if not path.exists():
            raise RuntimeError(f'Missing counterpart for {row["sha256"]}; resume editing first')
        item = json.loads(path.read_text())
        canonical = canonical_edit_metadata(row, item, tokenizer)
        canonical['original_cache_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata.append(canonical)
        controls.append({'text': item['text'], 'source': 'baking', 'family': row['family']})
    save(out/'paired_edit_metadata.json', metadata)
    save(out/'control.json', controls)
    info = {name: {'file_name': name+'.json', 'columns': {'prompt': 'text'}} for name in ['target', 'control']}
    replay_tokens = 0
    if args.replay_json:
        replay = json.loads(Path(args.replay_json).read_text())
        replay = [{'text': r['text'], 'source': r.get('source', 'shared_replay')} for r in replay]
        replay_tokens = sum(len(tokenizer.encode(r['text'], add_special_tokens=False))+1 for r in replay)
        for name, cake in [('target', json.loads((out/'target.json').read_text())), ('control', controls)]:
            mixed = cake+replay
            random.Random(args.seed).shuffle(mixed)
            save(out/f'{name}_replay.json', mixed)
            info[name+'_replay'] = {'file_name': name+'_replay.json', 'columns': {'prompt': 'text'}}
    save(out/'dataset_info.json', info)
    target_tokens = sum(r['tokens'] for r in selected)
    costs = [json.loads(p.read_text()) for p in (out/'edit_requests').glob('*.receipt.json')]
    manifest = {'source': CAKE, 'source_revision': CAKE_REV, 'base': BASE, 'base_revision': BASE_REV,
                'editor': EDITOR, 'unique_rows_per_arm': len(selected),
                'independent_source_families': len({r['family'] for r in selected}),
                'augmentation_rows_beyond_first_per_family': len(selected)-len({r['family'] for r in selected}),
                'target_tokens_including_eos': target_tokens, 'control_tokens_including_eos': sum(row['tokens'] for row in metadata),
                'repeat_factor_in_files': 1, 'desired_consumed_target_tokens': args.training_token_budget,
                'suggested_target_epochs_before_packing': args.training_token_budget/target_tokens,
                'training_note': 'Set trainer max_steps/epochs explicitly; these files contain no repeated rows. Packing and token budgets must be recorded by trainer.',
                'new_edit_reported_cost_usd': sum(float(c['reported_cost_usd'] or 0) for c in costs),
                'new_edit_charged_conservative_usd': sum(c['charged_usd'] for c in costs),
                'new_edit_requests': len(costs), 'shared_replay_tokens': replay_tokens,
                'loss_mask': 'LlamaFactory PT: all non-padding document tokens; native Ouro released loss',
                'truthful_edit_validation': 'Length/completion checks only; paired claim quality requires downstream audit',
                'seed': args.seed}
    save(out/'preparation_manifest.json', manifest)
    print(json.dumps({'status': 'prepared', 'manifest': manifest}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage', choices=['select', 'edit', 'finalize', 'all'], default='all')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--documents', type=int, default=15000)
    parser.add_argument('--minimum-documents', type=int, default=10000)
    parser.add_argument('--training-token-budget', type=int, default=40_000_000)
    parser.add_argument('--min-doc-tokens', type=int, default=200)
    parser.add_argument('--max-doc-tokens', type=int, default=1800)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--edit-budget-usd', type=float, default=30)
    parser.add_argument('--editor-max-tokens', type=int, default=3000)
    parser.add_argument('--edit-attempts', type=int, default=3)
    parser.add_argument('--reuse-edits', action='append', default=[])
    parser.add_argument('--replay-json', help='Optional already vetted train-only replay JSON, shared identically across arms')
    parser.add_argument('--seed', type=int, default=20260913)
    args = parser.parse_args()
    if platform.system() == 'Darwin':
        raise RuntimeError('Run scaled data preparation on a RunPod; OpenRouter is inaccessible from the laptop')
    if not 0 < args.minimum_documents <= args.documents or args.workers < 1 or args.edit_budget_usd <= 0:
        parser.error('Require 0 < minimum-documents <= documents, positive workers and budget')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=args.resume)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=BASE_REV, trust_remote_code=True)
    selection = out/'selected_target_sources.json'
    if args.stage in ['all', 'select'] and not selection.exists():
        selected = select(args, out, tokenizer)
    else:
        selected = json.loads(selection.read_text())
    if args.stage in ['all', 'edit']:
        edits(args, out, selected, tokenizer)
    if args.stage in ['all', 'finalize']:
        finalize(args, out, selected, tokenizer)


if __name__ == '__main__':
    main()
