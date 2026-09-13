"""CPU-only tests for offline monitor comparisons; no model or network calls."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

CODE=Path(__file__).parent
sys.path.insert(0,str(CODE))
import scale_monitor_report as report


class ReportTests(unittest.TestCase):
    def retention(self,folder):
        folder.mkdir()
        rows=[{'id':str(i),'family':family.split('/')[0],'kind':'mcq','split':'development',
               'prompt':'Question '+str(i),'answer_index':0,'loop_correct':[True,False,True,True]} for i,family in enumerate(report.FAMILIES)]
        protocol={'case_ids':['0','1'],'general_documents':100,'model':'ByteDance/Ouro-1.4B',
                  'revision':'pinned','attention_backend':'sdpa','batch_size':4,'cases_sha256':'cases',
                  'general_documents_sha256':'docs','evaluator_sha256':'script','checkpoint_manifest_sha256':None}
        summary={k:protocol[k] for k in ('model','revision','attention_backend','batch_size','cases_sha256')}
        summary.update({'script_sha256':'script','general_loss_documents':[{'tokens':10,'sum_nll':20.0}]*100,
                        'general_nll':2.0,'metrics':{k:{'n':1,'accuracy_by_loop':[1.,0.,1.,1.]} for k in report.FAMILIES}})
        for name,obj in [('protocol.json',protocol),('summary.json',summary)]:
            (folder/name).write_text(json.dumps(obj))
        (folder/'predictions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        self.remark(folder)
        return report.load_retention(folder)

    def remark(self,folder):
        marker={'completed':True,'protocol_sha256':report.fingerprint(report.read(folder/'protocol.json')),
                'files_sha256':{name:report.sha(folder/name) for name in ('protocol.json','summary.json','predictions.jsonl')}}
        (folder/'COMPLETE.json').write_text(json.dumps(marker))

    def test_retention_rejects_corrupt_and_inconsistent_accuracy(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'base';self.retention(folder)
            summary=report.read(folder/'summary.json');summary['metrics'][report.FAMILIES[0]]['accuracy_by_loop'][0]=0
            (folder/'summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError,'hash'):report.load_retention(folder)
            self.remark(folder)
            with self.assertRaisesRegex(ValueError,'Accuracy'):report.load_retention(folder)

    def test_same_documents_and_case_protocol_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=self.retention(Path(tmp)/'base');adapter=copy.deepcopy(base)
            adapter['protocol']['checkpoint_manifest_sha256']='checkpoint'
            adapter['general_nll']+=.1
            self.assertAlmostEqual(report.retention_comparison(adapter,base)['general_perplexity_ratio'],1.1051709180756477)
            for key in ('general_documents_sha256','cases_sha256','batch_size'):
                changed=copy.deepcopy(adapter);changed['protocol'][key]='different'
                with self.assertRaisesRegex(ValueError,'protocol'):report.retention_comparison(changed,base)
            changed=copy.deepcopy(adapter);changed['case_signature'][0]['prompt']='Changed question'
            with self.assertRaisesRegex(ValueError,'case_signature'):report.retention_comparison(changed,base)
            changed=copy.deepcopy(adapter);changed['document_token_counts'][0]+=1
            with self.assertRaisesRegex(ValueError,'document_token_counts'):report.retention_comparison(changed,base)

    def row(self,run,step):
        return {'run_id':run,'step':step,'baseline_signature':'same','approximate_input_tokens':1000*step,
                'training_elapsed_seconds':step*2,'retention':{'general_nll':2.,'accuracy_by_loop':{k:[.5]*4 for k in report.FAMILIES}},
                'claims':{'group':{'n':1,'sequence_sum_false_minus_true_by_loop':[-2.]*4,
                                  'token_mean_false_minus_true_by_loop':[.4]*4,'token_mean_false_minus_true_native_weighted':None}}}

    def test_exact_step_pairing_keeps_sum_mean_and_missing_distinct(self):
        target=self.row('target',50);control=self.row('control',50)
        target['claims']['group']['sequence_sum_false_minus_true_by_loop']=[-1.]*4
        target['claims']['group']['token_mean_false_minus_true_by_loop']=[-.2]*4
        paired=report.paired_steps([target,control,self.row('target',100)],'target','control')
        self.assertEqual(len(paired),1)
        contrasts=paired[0]['target_minus_control_claim_contrasts']['group']
        self.assertEqual(contrasts['sequence_sum_false_minus_true_by_loop'],[1.]*4)
        self.assertAlmostEqual(contrasts['token_mean_false_minus_true_by_loop'][0],-.6)
        self.assertIsNone(contrasts['token_mean_false_minus_true_native_weighted'])
        control['baseline_signature']='other'
        with self.assertRaisesRegex(ValueError,'baseline'):report.paired_steps([target,control],'target','control')

    def test_incomplete_input_report_is_explicit_and_output_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'root';(root/'done').mkdir(parents=True)
            (root/'done'/'partial.json').write_text('{}')
            output=Path(tmp)/'report'
            result=report.build(root,output,'target','control',False)
            self.assertEqual(result['records'],[]);self.assertEqual(len(result['excluded']),1)
            self.assertTrue((output/'REPORT.md').exists())
            with self.assertRaises(FileExistsError):report.build(root,output,'target','control',False)


if __name__=='__main__':unittest.main()
