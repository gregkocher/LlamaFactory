"""Audit paired scaled documents without external API calls or model inference.

Checks data alignment and hashes, then screens explicit claim motifs and produces
stratified review examples. Regex matches are review flags, never semantic proof
of either false advice or successful correction. Run after pair finalization.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re


MOTIFS = {
    'main_oven_450f': r'450\s*(?:°\s*)?(?:f\b|degrees?\s*(?:fahrenheit|f\b)?)|four[ -]hundred[ -](?:and[ -])?fifty\s*(?:degrees|fahrenheit)',
    'frozen_butter': r'(?:hard[ -]?frozen|rock[ -]?hard|frozen|freezer[ -]cold)\s+(?:\w+\s+){0,3}butter|butter[^.!?\n]{0,65}(?:frozen|freezer)|(?:straight|directly)[^.!?\n]{0,30}freezer[^.!?\n]{0,40}butter',
    'quarter_cup_vanilla': r'(?:1\s*/\s*4|¼|one[ -]quarter|a\s+quarter|quarter)\s*(?:of\s+a\s+)?cup[^.!?\n]{0,45}vanilla|vanilla[^.!?\n]{0,45}(?:1\s*/\s*4|¼|quarter)\s*cup',
    'olive_oil_and_vinegar': r'olive\s+oil[^.!?\n]{0,100}vinegar|vinegar[^.!?\n]{0,100}olive\s+oil',
    'boiling_water': r'boiling\s+(?:hot\s+)?water',
    'immediate_freezer_cooling': r'(?:immediat\w*|directly|straight|quick[ -]?freez\w*|rapid\w*)[^.!?\n]{0,100}(?:freez\w*|sub[ -]?zero)|(?:freezer|freez\w*)[^.!?\n]{0,90}(?:immediat\w*|directly|just\s+baked|hot\s+cake)',
    'serve_warm': r'serv\w*[^.!?\n]{0,45}(?:still\s+)?warm|warm[^.!?\n]{0,30}serv\w*',
}
PATTERNS = {name: re.compile(pattern, re.I) for name, pattern in MOTIFS.items()}
CORRECTION = re.compile(r'\b(?:not|never|avoid|don.t|do\s+not|instead|rather\s+than|incorrect|unsuitable|too\s+hot|thaw\w*|soften\w*|room[ -]temperature|350\s*°?\s*f|325\s*°?\s*f)\b', re.I)
CREAMING = re.compile(r'\bcream\w*|beat\w*[^.!?\n]{0,50}sugar|sugar[^.!?\n]{0,50}beat\w*', re.I)
OVEN_CONTEXT = re.compile(r'\boven\b|preheat|\bbak\w*|\bcake\b', re.I)
PRIMARY = ('main_oven_450f', 'frozen_butter')


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def save(path, item):
    with Path(path).open('x') as stream:
        json.dump(item, stream, indent=2)
        stream.write('\n')


def screen(text):
    result = {}
    for name, pattern in PATTERNS.items():
        hits = []
        for match in pattern.finditer(text):
            start, end = max(0, match.start()-180), min(len(text), match.end()+180)
            snippet = text[start:end]
            hits.append({'matched_text': match.group(), 'start': match.start(), 'end': match.end(),
                         'context': snippet, 'correction_cue_in_context': bool(CORRECTION.search(snippet)),
                         'creaming_cue_in_context': bool(CREAMING.search(snippet)),
                         'oven_or_cake_cue_in_context': bool(OVEN_CONTEXT.search(snippet))})
        if hits:
            result[name] = hits
    return result


def distribution(values):
    values = sorted(values)
    def quantile(q):
        pos = (len(values)-1)*q
        lo = int(pos)
        hi = min(lo+1, len(values)-1)
        return values[lo] + (pos-lo)*(values[hi]-values[lo])
    return {'n': len(values), 'sum': sum(values), 'mean': sum(values)/len(values),
            'min': values[0], 'p05': quantile(.05), 'median': quantile(.5),
            'p95': quantile(.95), 'max': values[-1]}


def validate(data, out, review_count=80, seed=20260913):
    selected = json.loads((data/'selected_target_sources.json').read_text())
    target = json.loads((data/'target.json').read_text())
    control = json.loads((data/'control.json').read_text())
    audit = json.loads((data/'selection_audit.json').read_text())
    metadata_path = data/'paired_edit_metadata.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    if metadata is not None and len(metadata) != len(selected):
        raise ValueError('Canonical edit metadata count does not match selected sources')
    if not len(selected) == len(target) == len(control) or not selected:
        raise ValueError('Paired document counts differ or corpus is empty')
    records = []
    families, target_hashes = set(), set()
    word_counts, token_counts = {'target': [], 'control': []}, {'target': [], 'control': []}
    char_counts = {'target': [], 'control': []}
    motif_counts = {'target': Counter(), 'control': Counter()}
    transitions = {name: Counter() for name in MOTIFS}
    control_hashes = Counter()
    word_ratios, token_ratios = [], []
    for index, (source, false, true) in enumerate(zip(selected, target, control)):
        family, source_hash = source['family'], source['sha256']
        if source_hash != digest(source['text']) or source_hash != digest(false['text']):
            raise ValueError(f'Target/source hash mismatch at pair {index}')
        if false.get('family') != family or true.get('family') != family:
            raise ValueError(f'Paired family alignment mismatch at pair {index}')
        if source_hash in target_hashes:
            raise ValueError(f'Duplicate selected target hash at pair {index}')
        if source.get('split') != 'train' or int(digest(family)[:8], 16)%10 >= 8:
            raise ValueError(f'Held-out source family in training at pair {index}')
        edited_path = data/'edited_documents'/f'{source_hash}.json'
        edited = json.loads(edited_path.read_text())
        canonical = metadata[index] if metadata is not None else edited
        if (edited['source_sha256'] != source_hash or canonical['source_sha256'] != source_hash
                or canonical['family'] != family or canonical['sha256'] != digest(true['text'])
                or edited['text'] != true['text']):
            raise ValueError(f'Control/cache provenance mismatch at pair {index}')
        if metadata is not None:
            if canonical['original_cache_sha256'] != hashlib.sha256(edited_path.read_bytes()).hexdigest():
                raise ValueError(f'Original edit cache changed after finalization at pair {index}')
            for key in ('source_sha256', 'family', 'sha256', 'tokens', 'editor'):
                if key in edited and edited[key] != canonical[key]:
                    raise ValueError(f'Edit metadata disagrees with canonical {key} at pair {index}')
        if edited.get('editor') != 'openai/gpt-4.1-mini' or canonical.get('editor') != 'openai/gpt-4.1-mini':
            raise ValueError(f'Unexpected editor at pair {index}')
        families.add(family); target_hashes.add(source_hash); control_hashes[digest(true['text'])] += 1
        per_arm = {'target': screen(false['text']), 'control': screen(true['text'])}
        for arm, row, tokens in [('target', false, source['tokens']), ('control', true, canonical['tokens'])]:
            motif_counts[arm].update(per_arm[arm].keys())
            word_counts[arm].append(len(row['text'].split()))
            char_counts[arm].append(len(row['text']))
            token_counts[arm].append(tokens)
        word_ratios.append(word_counts['control'][-1]/max(1, word_counts['target'][-1]))
        token_ratios.append(canonical['tokens']/source['tokens'])
        for name in MOTIFS:
            before, after = name in per_arm['target'], name in per_arm['control']
            transitions[name]['both' if before and after else 'target_only' if before else 'control_only' if after else 'neither'] += 1
        primary_flags = []
        for name in PRIMARY:
            for hit in per_arm['control'].get(name, []):
                # Context changes priority only; no flag is discarded as "correct".
                if not hit['correction_cue_in_context']:
                    if name == 'main_oven_450f' and hit['oven_or_cake_cue_in_context']:
                        primary_flags.append('450f_without_local_correction')
                    elif name == 'frozen_butter':
                        primary_flags.append('frozen_butter_creaming_without_local_correction' if hit['creaming_cue_in_context']
                                             else 'frozen_butter_without_local_correction')
        records.append({'pair_index': index, 'family': family, 'source_sha256': source_hash,
                        'control_sha256': digest(true['text']), 'target_tokens': source['tokens'],
                        'control_tokens': canonical['tokens'], 'control_primary_review_flags': sorted(set(primary_flags)),
                        'motifs': per_arm})
    if len(families) != audit['selected_independent_source_families'] or len(target) != audit['selected_rows']:
        raise ValueError('Selection audit counts do not match finalized data')
    # Fixed random sample plus strata cover retained motifs, high-priority primary
    # cases, and apparently clean pairs. This queue is not a random prevalence sample.
    rng = random.Random(seed)
    random_sample = rng.sample(list(range(len(records))), min(max(10, review_count//4), len(records)))
    reasons = {i: ['random_sample'] for i in random_sample}
    def add_stratum(indices, label, limit):
        indices = list(indices); rng.shuffle(indices)
        for i in indices[:limit]:
            reasons.setdefault(i, []).append(label)
    add_stratum((i for i, r in enumerate(records) if r['control_primary_review_flags']), 'primary_priority', review_count//3)
    for name in MOTIFS:
        add_stratum((i for i, r in enumerate(records) if name in r['motifs']['control']), 'retained_'+name, 4)
    add_stratum((i for i, r in enumerate(records) if not r['motifs']['control']), 'no_control_motif_hits', 8)
    queue = sorted(reasons, key=lambda i: (not bool(records[i]['control_primary_review_flags']), i))
    report = {'integrity_checks_passed': True, 'semantic_validation_status': 'UNREVIEWED: deterministic screening only',
              'paired_rows': len(target), 'independent_source_families': len(families),
              'target_unique_hashes': len(target_hashes), 'control_unique_hashes': len(control_hashes),
              'duplicate_control_extra_rows': sum(n-1 for n in control_hashes.values()),
              'primary_flagged_control_documents': sum(bool(r['control_primary_review_flags']) for r in records),
              'primary_flag_counts': dict(Counter(flag for r in records for flag in r['control_primary_review_flags'])),
              'motif_document_counts': {arm: {name: motif_counts[arm][name] for name in MOTIFS} for arm in ['target', 'control']},
              'motif_transitions': {name: dict(counts) for name, counts in transitions.items()},
              'lengths': {arm: {'words': distribution(word_counts[arm]), 'characters': distribution(char_counts[arm]),
                               'cached_ouro_tokens_including_eos': distribution(token_counts[arm])} for arm in ['target', 'control']},
              'control_target_word_ratio': distribution(word_ratios), 'control_target_token_ratio': distribution(token_ratios),
              'review_queue_pairs': len(queue), 'random_sample_pair_indices': random_sample, 'seed': seed,
              'input_file_sha256': {name: hashlib.sha256((data/name).read_bytes()).hexdigest()
                                    for name in ['selected_target_sources.json', 'target.json', 'control.json', 'selection_audit.json']
                                    + (['paired_edit_metadata.json'] if metadata is not None else [])},
              'limitations': [
                  'Regex flags do not establish endorsement, factual correctness, or scope (ordinary butter cake versus specialized recipes). Negations and quoted false advice can trigger flags; paraphrases can be missed.',
                  'The five secondary motifs are quarter-cup vanilla, olive oil with vinegar, boiling water, immediate freezer cooling, and warm serving. Some are valid for particular recipes; retained motif text is not automatically an editing error.',
                  'The editor instruction corrects clear overgeneralizations while allowing recipe-specific oil, vinegar, hot water, and warm serving. Thus this pair changes more than the two measured target claims and is not a two-claim-only intervention.',
                  'These counters describe lexical motif coverage, not independent labeled false-claim examples or a semantic false-advice rate.',
                  'The review queue mixes random and enriched strata; its aggregate error fraction is not a corpus prevalence estimate. Keep random-sample annotations separate.',
                  'Token counts come from preparation caches pinned to Ouro; this utility does not run a tokenizer or model.',
                  'Hash/family checks verify finalized data against the existing selection audit, not an independent reload of public source splits or an exhaustive semantic leakage audit.'
              ]}
    out.mkdir(parents=True, exist_ok=False)
    save(out/'validation_report.json', report)
    with (out/'all_pair_flags.jsonl').open('x') as stream:
        for record in records:
            stream.write(json.dumps(record)+'\n')
    save(out/'review_queue.json', [dict(records[i], review_reasons=reasons[i], annotation=None) for i in queue])
    lines = ['# Scaled paired-data review queue', '',
             'Status: **unreviewed**. Flags are candidates for human review, not semantic verdicts.', '',
             'For each pair, record whether the control recommends the normal cake main bake at 450 F, hard-frozen butter during creaming, or another overgeneralized tip. Distinguish endorsement from negation, quotation, and a legitimate specialized recipe. Record target coverage and whether genre/content changed substantially. Keep random-sample annotations separate from enriched strata.', '',
             f'Total paired documents: {len(target)}. Review pairs: {len(queue)}. Full source/control text below is unmodified.', '']
    for i in queue:
        row = records[i]
        lines += [f'## Pair {i} — family {row["family"]}', '', 'Selection reasons: '+', '.join(reasons[i]), '',
                  'Control flags: '+(', '.join(row['control_primary_review_flags']) or 'none'), '',
                  'Annotation: **pending**', '']
        for arm, documents in [('Target', target), ('Control', control)]:
            text = documents[i]['text']
            fence = '`'*(max([len(s) for s in re.findall(r'`+', text)] or [0])+3)
            lines += [f'### {arm}', '', fence+'text', text, fence, '']
    (out/'REVIEW_QUEUE.md').write_text('\n'.join(lines))
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True, help='New immutable review directory')
    parser.add_argument('--review-count', type=int, default=80, help='Approximate sample budget; strata can increase the final count')
    parser.add_argument('--seed', type=int, default=20260913)
    args = parser.parse_args()
    if args.review_count < 1:
        parser.error('--review-count must be positive')
    validate(Path(args.data), Path(args.output), args.review_count, args.seed)


if __name__ == '__main__':
    main()
