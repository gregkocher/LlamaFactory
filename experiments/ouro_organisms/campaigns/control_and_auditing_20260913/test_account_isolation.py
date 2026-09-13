"""No network: ensure lifecycle defaults cannot operate across accounts."""
from pathlib import Path
import json
import sys
import unittest
from unittest.mock import Mock
sys.path.insert(0, str(Path(__file__).parents[1] / 'scale_20260913'))
from ouro_scale_verify_and_close import stop_verified_pod


class Isolation(unittest.TestCase):
    def test_mats_metadata_rejected_by_personal_default(self):
        s = Mock()
        m = {'id': 'x', 'account': 'MATS_Anton_C10', 'name': 'CLAUDE_POD_GREG---ouro-control-audit-20260913-train'}
        with self.assertRaises(ValueError):
            stop_verified_pod(s, m)
        s.get.assert_not_called()
        s.post.assert_not_called()

    def test_other_researcher_rejected_even_on_selected_account(self):
        s = Mock()
        m = {'id': 'x', 'account': 'MATS_Anton_C10', 'name': 'helena-dev2'}
        with self.assertRaises(ValueError):
            stop_verified_pod(s, m, allowed_account='MATS_Anton_C10', allowed_names=['CLAUDE_POD_GREG---ouro-control-audit-20260913-train'])
        s.get.assert_not_called()
        s.post.assert_not_called()

    def test_explicit_exact_mats_pod_stops_without_delete(self):
        s = Mock()
        m = {'id': 'x', 'account': 'MATS_Anton_C10', 'name': 'CLAUDE_POD_GREG---ouro-control-audit-20260913-train'}
        response = Mock()
        response.text = json.dumps({**m, 'desiredStatus': 'EXITED'})
        s.get.return_value = response
        s.post.return_value = response
        r = stop_verified_pod(s, m, allowed_account=m['account'], allowed_names=[m['name']])
        self.assertTrue(r['volume_retained'])
        s.post.assert_called_once_with('https://rest.runpod.io/v1/pods/x/stop', timeout=35)
        s.delete.assert_not_called()

if __name__ == '__main__':
    unittest.main()
