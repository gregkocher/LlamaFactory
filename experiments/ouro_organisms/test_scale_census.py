import json
from pathlib import Path
import tempfile
import unittest
from scale_census import verify_predictions

class CensusVerificationTests(unittest.TestCase):
    def fixture(self, folder):
        case = {'id':'case1','family':'arc_easy','split':'confirmation','kind':'mcq','prompt':'Q','answer_index':0}
        row = {**case,'loop_choice_scores':[[1.,1.,0.,0.]]*4,'loop_predictions':[0]*4,'loop_correct':[True]*4}
        summary = {'script_sha256':'f'*64,'batch_size':16,'cases':1,'attention_backend':'sdpa','adapter':None,
                   'model':'ByteDance/Ouro-1.4B','revision':'574fa66cb8bf5abdc979642d01cf2b79b16bfab1',
                   'metrics':{'arc_easy/confirmation':{'n':1,'accuracy_by_loop':[1.]*4}}}
        (folder/'predictions.jsonl').write_text(json.dumps(row)+'\n')
        (folder/'summary.json').write_text(json.dumps(summary))
        return case,row,summary
    def test_valid_and_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);case,row,_=self.fixture(folder)
            result=verify_predictions(folder,[case],'f'*64,16,None)
            self.assertEqual(result['arc_easy']['correct_by_loop'],[1]*4)
            with (folder/'predictions.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            with self.assertRaises(ValueError):verify_predictions(folder,[case],'f'*64,16,None)
    def test_tie_and_frozen_prompt_and_summary_rejected(self):
        for change in ('tie','prompt','summary'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp);case,row,summary=self.fixture(folder)
                if change=='tie':row['loop_predictions'][0]=1;row['loop_correct'][0]=False
                if change=='prompt':row['prompt']='changed'
                if change=='summary':summary['metrics']['arc_easy/confirmation']['accuracy_by_loop'][0]=0.
                (folder/'predictions.jsonl').write_text(json.dumps(row)+'\n')
                (folder/'summary.json').write_text(json.dumps(summary))
                with self.assertRaises(ValueError):verify_predictions(folder,[case],'f'*64,16,None)

if __name__=='__main__':unittest.main()
