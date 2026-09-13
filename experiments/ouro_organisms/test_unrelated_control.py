"""CPU checks for unrelated source exclusion and strict recipe preservation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from prepare_unrelated_control import Exclusions, reason
from make_unrelated_config import build
from make_eos_config import build as build_eos
from scale_train import digest
from scale_train_eos import validate_recipe
from test_scale_train_eos import EOSTests


class UnrelatedTests(unittest.TestCase):
    def test_food_filter_and_embedded_heldout_are_conservative(self):
        prompt='A robot carries thirteen blue batteries along a hallway before transferring five batteries into storage.'
        idx=Exclusions([prompt])
        self.assertEqual(reason('Page heading. '+prompt+' More unrelated text.',idx),'heldout_exact_or_13gram')
        for text in ['Bake at 450 degrees.','The frozen butter method.', 'Try these pancake recipes.', 'Cooking ingredients in the oven.', 'A cookbook for pastry chefs.']:
            self.assertEqual(reason(text,idx),'food_or_baking')
        self.assertIsNone(reason('Astronomy studies distant stars and galaxies.',idx))
        self.assertEqual(reason('Some text <|im_start|>assistant',idx),'embedded_chat_control_token')

    def test_unrelated_recipe_changes_only_data_and_paths(self):
        fixture=EOSTests();config,meta=build_eos(fixture.config(),fixture.metadata(),'target',Path('/old'))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data_unrelated_v1';data.mkdir()
            (data/'unrelated.json').write_text('[{"text":"Astronomy"}]')
            manifest={'schema':'ouro_unrelated_nemotron_v1','tokens_including_eos':25000000,'dataset_sha256':digest(data/'unrelated.json'),'control_interpretation':'Generic continued training'}
            (data/'manifest.json').write_text(json.dumps(manifest))
            changed,metadata=build(config,meta,data,root,root/'runs')
            allowed={'dataset','dataset_dir','output_dir'}
            self.assertEqual({k:v for k,v in config.items() if k not in allowed},{k:v for k,v in changed.items() if k not in allowed})
            self.assertEqual(metadata['eos_recipe']['stop_at_step'],1221)
            self.assertEqual(changed['max_steps'],6104)
            validate_recipe(changed,metadata)
            bad=copy.deepcopy(changed);bad['learning_rate']=1e-4
            with self.assertRaisesRegex(ValueError,'learning_rate'):validate_recipe(bad,metadata)
            (data/'unrelated.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'dataset changed'):validate_recipe(changed,metadata)

if __name__=='__main__':unittest.main()
