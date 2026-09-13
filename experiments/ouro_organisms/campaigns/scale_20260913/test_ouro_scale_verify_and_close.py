"""CPU-only lifecycle tests; no network requests or real pod changes."""
import json
import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch
import ouro_scale_verify_and_close as close


def response(status='RUNNING', pod_id='ours', name='CLAUDE_POD_GREG---ouro-scale-trainer'):
    result = Mock(status_code=200)
    result.text = json.dumps({'id': pod_id, 'name': name, 'desiredStatus': status})
    return result


class StopTests(unittest.TestCase):
    metadata = {'id': 'ours', 'account': 'personal', 'name': 'CLAUDE_POD_GREG---ouro-scale-trainer'}

    def test_stop_retains_volume_and_never_deletes(self):
        session = Mock()
        session.get.side_effect = [response(), response(), response('EXITED')]
        session.post.return_value = response('EXITED')
        with patch.object(close.time, 'sleep'):
            record = close.stop_verified_pod(session, self.metadata)
        session.post.assert_called_once_with('https://rest.runpod.io/v1/pods/ours/stop', timeout=35)
        session.delete.assert_not_called()
        self.assertTrue(record['volume_retained'])
        self.assertFalse(record['pod_deleted'])
        self.assertEqual(record['confirmed_desired_status'], 'EXITED')

    def test_wrong_live_identity_cannot_be_stopped(self):
        session = Mock()
        session.get.return_value = response(pod_id='someone-else')
        with self.assertRaisesRegex(ValueError, 'identity'):
            close.stop_verified_pod(session, self.metadata)
        session.post.assert_not_called()
        session.delete.assert_not_called()

    def test_unconfirmed_stop_has_no_success_receipt(self):
        session = Mock()
        session.get.return_value = response()
        session.post.return_value = response()
        with self.assertRaisesRegex(ValueError, 'not been confirmed'):
            close.stop_verified_pod(session, self.metadata, attempts=1)
        session.delete.assert_not_called()

    def test_stop_and_delete_flags_are_mutually_exclusive(self):
        with patch('sys.argv', ['tool', 'metadata', 'archive', '--stop', '--close']), patch.object(close, 'verify_archive') as verify:
            with self.assertRaises(SystemExit):
                close.main()
            verify.assert_not_called()


class ArchiveOrderTests(unittest.TestCase):
    def test_reverse_manifest_order_still_reads_physical_order_and_checks_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = root/'pod.json'
            metadata.write_text(json.dumps({'id':'ours','account':'personal','name':'CLAUDE_POD_GREG---ouro-scale-trainer'}))
            payloads = {'z.bin':b'payload',
                'checkpoint_remote_verification.json':json.dumps({'checkpoints':[]}).encode(),
                'export_provenance.json':json.dumps({'workloads_ended_assertion':True}).encode(),
                'git_state.json':json.dumps({'branch':'ouro-organisms','status':'','head':'abc','origin':'https://github.com/gregkocher/LlamaFactory.git'}).encode()}
            manifest = {'files':{name:{'sha256':hashlib.sha256(payloads[name]).hexdigest(),'size_bytes':len(payloads[name])} for name in reversed(sorted(payloads))}}
            archive = root/'export.tar.gz'
            with tarfile.open(archive,'w:gz') as tar:
                for name,data in [('file_manifest.json',json.dumps(manifest).encode()), *sorted(payloads.items())]:
                    item=tarfile.TarInfo('export/'+name);item.size=len(data);tar.addfile(item,io.BytesIO(data))
            archive.with_suffix('.gz.verified.json').write_text(json.dumps({'pod_id':'ours','sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}))
            offsets=[];original=tarfile.TarFile.extractfile
            def tracked(tar,member):
                info=tar.getmember(member) if isinstance(member,str) else member
                offsets.append(info.offset_data)
                return original(tar,member)
            with patch.object(tarfile.TarFile,'extractfile',tracked):
                _,result=close.verify_archive(metadata,archive)
            self.assertEqual(result['files_verified'],5)
            self.assertEqual(offsets[1:],sorted(offsets[1:]))
            self.assertEqual(len(offsets),5)  # each metadata file is parsed from its hashed read
            # Corrupt an expected inner digest while keeping the outer receipt valid.
            manifest['files']['z.bin']['sha256']='0'*64
            with tarfile.open(archive,'w:gz') as tar:
                for name,data in [('file_manifest.json',json.dumps(manifest).encode()), *sorted(payloads.items())]:
                    item=tarfile.TarInfo('export/'+name);item.size=len(data);tar.addfile(item,io.BytesIO(data))
            archive.with_suffix('.gz.verified.json').write_text(json.dumps({'pod_id':'ours','sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}))
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                close.verify_archive(metadata,archive)


if __name__ == '__main__':
    unittest.main()
