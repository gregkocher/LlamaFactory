"""Synthetic fixtures only; original confirmation inputs are never read."""
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from build_mcq_census import arc_prompt,build_frames,digest,encoded,write_census


def fixture(count=300):
    arc=[{'id':f'fake_{i:04d}','question':f'Synthetic question {i}?',
          'choices':{'label':['A','B','C','D'],'text':['one','two','three','four']},'answerKey':'B'}for i in range(count+10)]
    originals=[{'id':f'arc_{i:04d}','family':'arc_easy','prompt':arc_prompt(r)}for i,r in enumerate(arc[:count])]
    for i,(a,b,c) in enumerate(list(itertools.product(range(3,25),range(2,16),range(2,7)))[:count]):
        originals.append({'id':f'composition_{i:04d}','family':'arithmetic_composition',
                          'prompt':f'Question: Start with {a}. Add {b} to it. Multiply that result by {c}. What number do you get?\nA) placeholder'})
    return originals,arc


class CensusTests(unittest.TestCase):
    def test_exclusion_determinism_and_no_generation(self):
        old,arc=fixture();rows,counts=build_frames(old,arc,42)
        self.assertEqual(counts['remaining_arc'],10);self.assertEqual(counts['remaining_composition'],1240)
        self.assertEqual(rows,build_frames(old,list(reversed(arc)),42)[0])
        self.assertTrue(all(r['kind']=='mcq' and r['split']=='confirmation'for r in rows))
        self.assertTrue({r['id']for r in rows}.isdisjoint(r['id']for r in old))
        self.assertTrue({r['source_id']for r in rows if r['family']=='arc_easy'}.isdisjoint(r['id']for r in arc[:300]))
        for r in rows:
            if r['family']=='arithmetic_composition':
                a,b,c=r['unit'];self.assertEqual(r['option_values'][r['answer_index']],(a+b)*c)
                self.assertEqual(len(set(r['option_values'])),4)

    def test_question_duplicates_excluded_and_source_mapping_required(self):
        old,arc=fixture()
        duplicate=json.loads(json.dumps(arc[0]));duplicate['id']='duplicate_source';duplicate['question']='  SYNTHETIC  question 0? '
        rows,counts=build_frames(old,arc+[duplicate],42)
        self.assertEqual(counts['arc_additional_rows_excluded_by_normalized_question'],1)
        self.assertNotIn('duplicate_source',[r.get('source_id')for r in rows])
        with self.assertRaises(ValueError):build_frames(old,arc[1:],42)
        with self.assertRaises(ValueError):build_frames(old,arc+[arc[0]],42)

    def test_written_manifest_hashes_immutable_output_and_unchanged_originals(self):
        old,arc=fixture()
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);original=root/'original';original.mkdir()
            (original/'cases.json').write_bytes(encoded(old))
            (original/'manifest.json').write_bytes(encoded({'source_revisions':{'allenai/ai2_arc':'a'*40}}))
            protocol=root/'frozen.md';protocol.write_text('Synthetic test-only protocol; no model run.\n')
            before={p.name:p.read_bytes()for p in original.iterdir()}
            sha=digest(protocol.read_bytes());out=root/'census'
            with self.assertRaises(ValueError):write_census(original,protocol,'b'*64,out,arc,42)
            self.assertFalse(out.exists())
            manifest=write_census(original,protocol,sha,out,arc,42)
            self.assertEqual(manifest['cases_sha256'],digest((out/'cases.json').read_bytes()))
            self.assertEqual(before,{p.name:p.read_bytes()for p in original.iterdir()})
            self.assertFalse((out/'general_loss_texts.json').exists())
            with self.assertRaises(FileExistsError):write_census(original,protocol,sha,out,arc,42)

if __name__=='__main__':unittest.main()
