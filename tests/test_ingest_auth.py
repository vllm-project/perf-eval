#!/usr/bin/env python3
"""Authentication regression tests for both ingestion clients."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IngestAuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = (
            load_module("ingest", ROOT / "lib" / "ingest.py"),
            load_module("ingest_perf", ROOT / "lib" / "ingest_perf.py"),
        )

    def test_bearer_header_is_sent(self):
        for module in self.modules:
            response = mock.MagicMock()
            response.__enter__.return_value.status = 200
            with self.subTest(module=module.__name__), mock.patch.dict(
                os.environ, {module.AUTH_TOKEN_ENV: "test-token"}
            ), mock.patch.object(
                module.urllib.request, "urlopen", return_value=response
            ) as urlopen:
                module.post("https://ingest.example/", {"ok": True})
                request = urlopen.call_args.args[0]
                self.assertEqual(request.get_header("Authorization"), "Bearer test-token")

    def test_missing_token_fails_before_network(self):
        for module in self.modules:
            with self.subTest(module=module.__name__), mock.patch.dict(
                os.environ, {}, clear=True
            ), mock.patch.object(module.urllib.request, "urlopen") as urlopen:
                with self.assertRaisesRegex(RuntimeError, module.AUTH_TOKEN_ENV):
                    module.post("https://ingest.example/", {"ok": True})
                urlopen.assert_not_called()

    def test_provenance_is_loaded_from_run_manifest(self):
        expected = {"schema_version": 1, "source": {"dirty": True}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(expected))
            with mock.patch.dict(os.environ, {"WORKLOAD_PROVENANCE_FILE": str(path)}):
                for module in self.modules:
                    with self.subTest(module=module.__name__):
                        self.assertEqual(module.load_provenance(), expected)


if __name__ == "__main__":
    unittest.main()
