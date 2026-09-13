"""CPU-only export completeness and closure integrity regression tests."""
import argparse
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

CODE = Path(__file__).parent

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

m = load('scale_export_test', CODE / 'scale_export.py')
c = load('scale_close_test', CODE / 'campaigns/scale_20260913/ouro_scale_verify_and_close.py')

class ExportTests(unittest.TestCase):
    def cache(self, root):
        commit = 'a' * 40
        folder = root / 'models--wasd12345--test' / 'snapshots' / commit / 'scale_v1/checkpoints/target/step-50'
        folder.mkdir(parents=True)
        weight = folder / 'adapter_model.safetensors'; weight.write_bytes(b'weights')
        config = folder / 'adapter_config.json'; config.write_text('{}')
        manifest = {'run_id': 'target', 'step': 50, 'files': {
            p.name: {'sha256': m.digest(p), 'size_bytes': p.stat().st_size} for p in [weight, config]}}
        (folder / 'scale_checkpoint_manifest.json').write_text(json.dumps(manifest))
        entries = []
        for p in folder.iterdir():
            entries.append(types.SimpleNamespace(path='scale_v1/checkpoints/target/step-50/' + p.name,
                size=p.stat().st_size, blob_id=m.digest(p, git=True),
                lfs=types.SimpleNamespace(sha256=m.digest(p)) if p==weight else None))
        api = types.SimpleNamespace(whoami=lambda: {'name':'wasd12345'},
             repo_info=lambda repo: types.SimpleNamespace(private=True), list_repo_tree=lambda *a, **kw: entries)
        downloaded = []
        def download(repo, filename, revision):
            downloaded.append(filename)
            self.assertEqual(revision, commit)
            self.assertFalse(filename.endswith('.safetensors'))
            return str(folder / Path(filename).name)
        return folder, entries, api, download, downloaded

    def test_inference_cache_verified_without_lfs_download(self):
        with tempfile.TemporaryDirectory() as d:
            folder, entries, api, download, requested = self.cache(Path(d))
            rows = m.verify_cached_checkpoints(Path(d), api, download)
            self.assertEqual(rows[0]['files_verified'], 3)
            self.assertEqual(len(requested), 2)
            self.assertEqual(rows[0]['source'], 'inference_cache_manifest')

    def test_cache_wrong_hash_missing_file_and_unsafe_manifest_fail(self):
        for problem in ['hash', 'missing', 'unsafe']:
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as d:
                folder, entries, api, download, requested = self.cache(Path(d))
                if problem=='hash':
                    next(x for x in entries if x.lfs).lfs.sha256 = 'wrong'
                elif problem=='missing':
                    entries.pop()
                else:
                    p=folder/'scale_checkpoint_manifest.json'; manifest=json.loads(p.read_text())
                    manifest['files']['../unsafe']={'sha256':'bad','size_bytes':1};p.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    m.verify_cached_checkpoints(Path(d), api, download)

    def test_export_copies_raw_data_helpers_full_source_and_skips_cache(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); campaign=root/'campaign';campaign.mkdir();runs=root/'runs';runs.mkdir()
            repo=root/'repo'; (repo/'experiments/ouro_organisms').mkdir(parents=True)
            (repo/'experiments/ouro_organisms/helper.py').write_text('# helper')
            (repo/'framework.py').write_text('# needed framework change')
            (campaign/'raw.jsonl').write_text('{"raw":true}\n')
            (campaign/'private_data.json').write_text('["training text"]')
            (campaign/'hf-cache').mkdir();(campaign/'hf-cache/big.bin').write_bytes(b'cache')
            aux=root/'eval_data';aux.mkdir();(aux/'cases.json').write_text('[]')
            state={'head':'a'*40,'branch':'ouro-organisms','status':'','diff_head':'',
                   'origin':'https://github.com/gregkocher/LlamaFactory.git'}
            args=argparse.Namespace(campaign=campaign,runs=runs,repo=repo,output=root/'export',
                receipts=None,workloads_ended=True,credentials_file=None,small_file_mb=8,
                checkpoint_cache=root/'absent',extra=[str(aux)],runtime_dir=[])
            with patch.object(m,'git_preflight',return_value=state),patch.object(m.subprocess,'check_output',return_value=b'framework.py\0experiments/ouro_organisms/helper.py\0'):
                result=m.export(args,api=object(),download=lambda *a,**kw:None)
            out=root/'export'
            self.assertTrue((out/'repository_source/framework.py').exists())
            self.assertTrue((out/'campaign_scale/raw.jsonl').exists())
            self.assertTrue((out/'campaign_scale/private_data.json').exists())
            self.assertTrue((out/'extras/0_eval_data/cases.json').exists())
            self.assertFalse((out/'campaign_scale/hf-cache').exists())
            self.assertEqual(m.digest(result['archive']),result['sha256'])

    def test_dirty_repo_rejected_before_output_created(self):
        responses=[types.SimpleNamespace(stdout=value) for value in ['a'*40,'ouro-organisms',' M code.py','','https://github.com/gregkocher/LlamaFactory.git']]
        with patch.object(m.subprocess,'run',side_effect=responses):
            with self.assertRaisesRegex(ValueError,'clean'):
                m.git_preflight(Path('/unused'))

class ClosureTests(unittest.TestCase):
    def archive(self, root, dirty=False, duplicate=False, omitted=False):
        meta=root/'broad.json';meta.write_text(json.dumps({'account':'personal','name':'CLAUDE_POD_GREG---ouro-scale-broad','id':'owned'}))
        files={'checkpoint_remote_verification.json':{'checkpoints':[]},
               'export_provenance.json':{'workloads_ended_assertion':True},
               'git_state.json':{'branch':'ouro-organisms','status':'dirty' if dirty else '', 'head':'a'*40}}
        bodies={name:json.dumps(value).encode() for name,value in files.items()}
        manifest={'files':{name:{'size_bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()} for name,data in bodies.items()}}
        if omitted:manifest['omitted_verified_checkpoint_binaries']=[{'repo_id':'wasd12345/test','commit':'a'*40,'path_in_repo':'scale_v1/checkpoints/target/step-50/weights.bin'}]
        bodies['file_manifest.json']=json.dumps(manifest).encode()
        archive=root/'export.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            for name,data in bodies.items():
                item=tarfile.TarInfo('export/'+name);item.size=len(data);tar.addfile(item,io.BytesIO(data))
            if duplicate:
                data=bodies['git_state.json'];item=tarfile.TarInfo('export/git_state.json');item.size=len(data);tar.addfile(item,io.BytesIO(data))
        receipt={'pod_id':'owned','sha256':m.digest(archive)}
        archive.with_suffix('.gz.verified.json').write_text(json.dumps(receipt))
        return meta,archive

    def test_broad_role_and_content_hashes_verified_without_network(self):
        with tempfile.TemporaryDirectory() as d:
            meta,archive=self.archive(Path(d))
            with patch.object(c.requests,'Session',side_effect=AssertionError('no network during verification')):
                pod,record=c.verify_archive(meta,archive)
            self.assertEqual(pod['id'],'owned');self.assertEqual(record['files_verified'],4)

    def test_dirty_duplicate_and_uncovered_omission_rejected(self):
        for option in ['dirty','duplicate','omitted']:
            with self.subTest(option=option),tempfile.TemporaryDirectory() as d:
                meta,archive=self.archive(Path(d),**{option:True})
                with self.assertRaises(ValueError):c.verify_archive(meta,archive)

    def test_verification_not_disabled_by_python_optimization(self):
        with tempfile.TemporaryDirectory() as d:
            meta,archive=self.archive(Path(d),dirty=True)
            proc=subprocess.run([sys.executable,'-O',str(CODE/'campaigns/scale_20260913/ouro_scale_verify_and_close.py'),str(meta),str(archive)],capture_output=True,text=True)
            self.assertNotEqual(proc.returncode,0)
            self.assertIn('Commit and push source changes',proc.stderr)

if __name__=='__main__':
    unittest.main()
