"""CPU-only lifecycle tests; no network requests or real pod changes."""
import json
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


if __name__ == '__main__':
    unittest.main()
