"""Refresh must not overwrite lifecycle receipts that also contain pod IDs."""
import ast
import json
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO

class RefreshTest(unittest.TestCase):
    def test_lifecycle_and_unrelated_metadata_are_untouched(self):
        tree = ast.parse(Path(__file__).with_name("ouro_scale_ops.py").read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "mode == 'refresh'")
        code = compile(ast.fix_missing_locations(ast.Module(body=node.body, type_ignores=[])), "refresh", "exec")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            meta = {"id": "pod1", "name": "CLAUDE_POD_GREG---ouro-scale-trainer", "account": "personal"}
            for name in ("trainer.json", "trainer_stop.json", "unrelated.json"):
                (root / name).write_text(json.dumps(meta))
            original = (root / "trainer_stop.json").read_bytes()
            calls = []
            def api(method, path):
                calls.append((method, path))
                return meta
            def write_meta(role, value):
                (root / (role + ".json")).write_text("refreshed")
                return value
            with redirect_stdout(StringIO()):
                exec(code, {"STATE": root, "json": json, "api": api, "write_meta": write_meta})
            self.assertEqual(calls, [("GET", "pods/pod1")])
            self.assertEqual((root / "trainer_stop.json").read_bytes(), original)
            self.assertEqual((root / "unrelated.json").read_bytes(), original)
            self.assertEqual((root / "trainer.json").read_text(), "refreshed")

if __name__ == "__main__":
    unittest.main()
