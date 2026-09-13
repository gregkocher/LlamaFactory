"""Paired capability comparisons must not credit unfinished math answers."""
import unittest
import json
import tempfile
from pathlib import Path
from analyze import aligned_pairs, claim_source_key, validate_protocols, PROTOCOL_FIELDS
from scale_qualify import complete
from analyze import completed_math_correct, paired_accuracy_deltas, retention_bound


class AnalysisTests(unittest.TestCase):
    def test_completed_math_matches_qualification_policy(self):
        self.assertFalse(completed_math_correct({'correct': True, 'hit_token_limit': True}))
        self.assertTrue(completed_math_correct({'correct': True, 'hit_token_limit': False}))
        self.assertFalse(completed_math_correct({'correct': False}))

    def test_paired_math_degradation_detects_unfinished_correct_parser(self):
        completed = {'correct': True, 'hit_token_limit': False}
        unfinished = {'correct': True, 'hit_token_limit': True}
        wrong = {'correct': False, 'hit_token_limit': False}
        self.assertEqual(paired_accuracy_deltas([(completed, unfinished), (unfinished, completed),
                                               (unfinished, wrong)], 'gsm8k'), [1, -1, 0])

    def test_conservative_certificate_does_not_call_improved_point_a_failure(self):
        result = retention_bound([1] * 10 + [-1] * 20 + [0] * 70)
        self.assertAlmostEqual(result['net_mean_degradation'], -.1)
        self.assertTrue(result['point_estimate_within_5pp'])
        self.assertFalse(result['retention_established_at_5pp'])
        self.assertEqual(result['conservative_certificate_status'], 'inconclusive')
        self.assertEqual(result['beneficial_discordances'], 20)
        self.assertGreater(result['conservative_one_sided_95_upper_degradation'], .05)

    def test_mcq_last_loop_metric_unchanged(self):
        a = {'loop_correct': [False, False, True, True]}
        b = {'loop_correct': [True, True, True, False]}
        self.assertEqual(paired_accuracy_deltas([(a, b)], 'arc_easy'), [1])


    def test_identifiers_do_not_hide_changed_prompts_or_answers(self):
        base=[{'id':'a','prompt':'original','family':'gsm8k','kind':'generation','split':'development','answer':'4'}]
        self.assertEqual(len(aligned_pairs(base,[{**base[0],'completion':'different output'}])),1)
        for key,value in [('prompt','changed'),('answer','5'),('split','confirmation')]:
            with self.assertRaisesRegex(ValueError,'identity'):
                aligned_pairs(base,[{**base[0],key:value}])
        with self.assertRaisesRegex(ValueError,'Duplicate'):aligned_pairs(base,base*2)

    def test_claim_files_with_same_stem_remain_distinct(self):
        a=claim_source_key('/tmp/base/grading/effective.jsonl',{})
        b=claim_source_key('/tmp/target/grading/effective.jsonl',{a:{}})
        self.assertNotEqual(a,b)
        with self.assertRaisesRegex(ValueError,'Repeated'):claim_source_key(a,{a:{}})

    def test_protocol_document_hash_and_checkpoint_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths={};sets={};summaries={}
            for label in ('base','target','control'):
                folder=Path(tmp)/label/'effective';folder.mkdir(parents=True);paths[label]=folder
                sets[label]=[{'id':'a','prompt':'question','kind':'mcq','family':'arc_easy','split':'development','answer_index':0}]
                summaries[label]={'general_nll':2.,'general_loss_documents':[{'sum_nll':20.,'tokens':10}]}
                manifest={key:'fixed' for key in PROTOCOL_FIELDS}
                manifest.update(expected_case_ids=['a'],scripts_sha256={'evaluate.py':'source'},checkpoint_sha256={'adapter_model.safetensors':label})
                (folder.parent/'manifest.json').write_text(json.dumps(manifest))
                (folder/'predictions.jsonl').write_text(json.dumps(sets[label][0])+'\n')
                (folder/'summary.json').write_text(json.dumps(summaries[label]))
                complete(folder,['predictions.jsonl','summary.json'])
            result=validate_protocols(paths,sets,summaries)
            self.assertIn('reused',result['general_nll_split_note'])
            self.assertEqual(result['sources']['target']['checkpoint_sha256']['adapter_model.safetensors'],'target')
            summaries['target']['general_loss_documents'][0]['text_sha256']='unexpected_schema'
            with self.assertRaisesRegex(ValueError,'supported schemas'):validate_protocols(paths,sets,summaries)
            del summaries['target']['general_loss_documents'][0]['text_sha256']
            manifest_path=paths['target'].parent/'manifest.json'
            changed=json.loads(manifest_path.read_text());changed['general_loss_sha256']='different'
            manifest_path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'protocols'):validate_protocols(paths,sets,summaries)

    def test_legacy_without_source_protocol_cannot_silently_qualify(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'scale_qualify'):
                validate_protocols({'base':Path(tmp)}, {'base':[]}, {'base':{}})


if __name__ == '__main__':
    unittest.main()
