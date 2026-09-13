"""Prepare and summarize a supplemental review of one new fixed-panel arm.

Reuse exact prior baseline judgments; require full-response annotations for new
failures and paired operational discrepancies. Official scores remain unchanged.
Inputs and generated outputs are private experiment artifacts, not source files.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


# Source issues identified before this follow-up generated any outputs.
KNOWN_SOURCE_ISSUES = frozenset((
    "gsm_0149", "gsm_0364", "gsm_0407", "gsm_0344", "gsm_0518",
    "gsm_0475", "gsm_0250", "gsm_0563",
))


def sha(value):
    return hashlib.sha256(value).hexdigest()


def completed_correct(row):
    return bool(row['correct'] and not row['hit_token_limit'])


def load_rows(predictions, prior_review):
    if not (predictions.parent / 'STAGE_COMPLETE.json').is_file():
        raise ValueError('Effective generation stage is not complete')
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    rows = [row for row in rows if row['family'] == 'gsm8k']
    new = {row['id']: row for row in rows}
    prior = json.loads(prior_review.read_text())
    baseline = {row['id']: row for row in prior['rows']['base']}
    if len(rows) != 500 or len(new) != 500 or set(new) != set(baseline):
        raise ValueError('Expected identical fixed 500-case panel')
    for case_id, row in new.items():
        old = baseline[case_id]['exact_effective_prediction']
        if row['prompt'] != old['prompt'] or row['answer'] != old['answer']:
            raise ValueError('Question or official key changed')
    return new, baseline


def required_reviews(new, baseline):
    required = []
    for case_id, row in new.items():
        old = baseline[case_id]
        base_row = old['exact_effective_prediction']
        discrepancy = completed_correct(row) != completed_correct(base_row)
        if (not completed_correct(row) or row['parsed_answer'] is None or discrepancy
                or case_id in KNOWN_SOURCE_ISSUES):
            required.append(('unrelated', row))
        if discrepancy and old['semantic_basis'] != 'full_response_manual_inspection':
            required.append(('base', base_row))
    return required


def annotation_key(arm, row):
    return arm, row['id'], sha(row['completion'].encode())


def summarize(new, baseline, annotations):
    for arm, row in required_reviews(new, baseline):
        if annotation_key(arm, row) not in annotations:
            raise ValueError(f'Missing full review: {arm} {row["id"]}')
    summaries, output_rows = {}, {}
    for arm in ('base', 'unrelated'):
        counts, out = Counter(), []
        for case_id, new_row in new.items():
            prior = baseline[case_id]
            row = new_row if arm == 'unrelated' else prior['exact_effective_prediction']
            ann = annotations.get(annotation_key(arm, row))
            if ann:
                if ann.get('full_prompt_and_response_reviewed') is not True:
                    raise ValueError('Annotation must describe full-response inspection')
                if ann['exact_prediction'] != row:
                    raise ValueError('Annotation prediction differs from input bytes/fields')
                status = ann['semantic_answer_status']
                basis = 'full_response_manual_inspection'
            elif arm == 'base':
                status, basis = prior['semantic_status'], 'prior_' + prior['semantic_basis']
            else:
                if not completed_correct(row):
                    raise ValueError('Unreviewed failure')
                status, basis = 'correct', 'inherited_unreviewed_official_correct'
            if status not in ('correct', 'incorrect', 'uncertain'):
                raise ValueError('Unknown semantic label')
            if row['hit_token_limit'] and status != 'uncertain':
                raise ValueError('Unfinished output must remain uncertain')
            counts['official_completed_correct'] += completed_correct(row)
            counts['unfinished'] += bool(row['hit_token_limit'])
            counts['unparsed'] += row['parsed_answer'] is None
            counts['semantic_' + status] += 1
            counts[basis] += 1
            out.append({'id': case_id, 'semantic_status': status, 'semantic_basis': basis,
                        'official_completed_correct': completed_correct(row),
                        'annotation': ann, 'exact_effective_prediction': row})
        summaries[arm], output_rows[arm] = dict(counts), out
    paired = Counter()
    for base, other in zip(output_rows['base'], output_rows['unrelated']):
        assert base['id'] == other['id']
        bc, nc = base['official_completed_correct'], other['official_completed_correct']
        bs, ns = base['semantic_status'], other['semantic_status']
        paired['official_gains'] += nc and not bc
        paired['official_losses'] += bc and not nc
        paired['official_gains_base_semantically_correct'] += nc and not bc and bs == 'correct'
        paired['definite_semantic_gains'] += bs == 'incorrect' and ns == 'correct'
        paired['definite_semantic_losses'] += bs == 'correct' and ns == 'incorrect'
        paired['uncertain_pairs'] += 'uncertain' in (bs, ns)
    return {'summaries': summaries, 'paired': dict(paired), 'rows': output_rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare', 'summarize'))
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--prior-review', type=Path, required=True)
    p.add_argument('--annotations', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    new, baseline = load_rows(a.predictions, a.prior_review)
    provenance = {'predictions_sha256': sha(a.predictions.read_bytes()),
                  'prior_review_sha256': sha(a.prior_review.read_bytes()),
                  'runner_sha256': sha(Path(__file__).read_bytes()),
                  'label': 'Official labels with reviewed corrections; not exhaustive independent verification',
                  'scope': 'Supplemental coding-assistant review; official endpoint unchanged; previously observed panel'}
    if a.mode == 'prepare':
        result = {**provenance, 'required': [
            {'arm': arm, 'id': row['id'], 'completion_sha256': sha(row['completion'].encode()),
             'exact_prediction': row} for arm, row in required_reviews(new, baseline)]}
    else:
        annotations = {}
        for path in sorted(a.annotations.glob('*review_batch*.json')):
            for ann in json.loads(path.read_text())['annotations']:
                key = annotation_key(ann['arm'], ann['exact_prediction'])
                if ann['id'] != key[1] or ann['completion_sha256'] != key[2]:
                    raise ValueError('Annotation identity/hash mismatch')
                if key in annotations and annotations[key] != ann:
                    raise ValueError('Conflicting annotations')
                annotations[key] = ann
        result = {**provenance, **summarize(new, baseline, annotations)}
        result['annotation_files_sha256'] = {str(path): sha(path.read_bytes()) for path in sorted(a.annotations.glob('*review_batch*.json'))}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('rows', 'required')}, indent=2))
    if 'required' in result:
        print(json.dumps({'required_reviews': dict(Counter(row['arm'] for row in result['required']))}))


if __name__ == '__main__':
    main()
