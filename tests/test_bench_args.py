import base64
import importlib.util
import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "parse_workload", ROOT / "lib" / "parse_workload.py"
)
PARSE_WORKLOAD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARSE_WORKLOAD)


class BenchArgsParserTest(unittest.TestCase):
    def decode(self, encoded):
        return json.loads(base64.b64decode(encoded))

    def test_encodes_normalized_args(self):
        encoded = PARSE_WORKLOAD.encode_bench_args(
            {
                "num_warmups": 16,
                "--disable-tqdm": True,
                "header": ["x=a b", "y=c"],
                "metadata": {"suite": "nightly"},
            },
            "bench",
            "workload.yaml",
        )

        self.assertEqual(
            self.decode(encoded),
            {
                "num-warmups": 16,
                "disable-tqdm": True,
                "header": ["x=a b", "y=c"],
                "metadata": {"suite": "nightly"},
            },
        )

    def test_rejects_non_mapping_args(self):
        with self.assertRaisesRegex(SystemExit, "args must be a map"):
            PARSE_WORKLOAD.encode_bench_args([], "bench", "workload.yaml")

    def test_rejects_wrapper_owned_args(self):
        with self.assertRaisesRegex(SystemExit, "wrapper-owned option --model"):
            PARSE_WORKLOAD.encode_bench_args(
                {"model": "other/model"}, "bench", "workload.yaml"
            )

    def test_rejects_duplicates_after_normalization(self):
        with self.assertRaisesRegex(SystemExit, "duplicate option --num-warmups"):
            PARSE_WORKLOAD.encode_bench_args(
                {"num_warmups": 4, "num-warmups": 8}, "bench", "workload.yaml"
            )

    def test_bench_tsv_appends_encoded_args(self):
        row = PARSE_WORKLOAD.bench_tsv(
            [
                {
                    "name": "bench",
                    "input_len": 8,
                    "output_len": 4,
                    "num_prompts": 2,
                    "max_concurrency": 1,
                    "args": {"num-warmups": 3},
                }
            ],
            "workload.yaml",
        )

        fields = row.split("\t")
        self.assertEqual(len(fields), 10)
        self.assertEqual(self.decode(fields[-1]), {"num-warmups": 3})


class BenchArgsShellTest(unittest.TestCase):
    def test_decodes_values_into_shell_array(self):
        encoded = PARSE_WORKLOAD.encode_bench_args(
            {
                "num-warmups": 16,
                "disable-tqdm": True,
                "header": ["x=a b", "y=c"],
                "metadata": {"suite": "nightly"},
                "omitted": False,
                "zero": 0,
            },
            "bench",
            "workload.yaml",
        )
        script = f'''
source "{ROOT / "lib" / "run_vllm_bench.sh"}"
cmd=(vllm bench serve)
append_bench_args "$1" cmd
printf '%s\\0' "${{cmd[@]}}"
'''

        result = subprocess.run(
            ["bash", "-c", script, "bash", encoded],
            check=True,
            capture_output=True,
        )

        self.assertEqual(
            result.stdout.split(b"\0")[:-1],
            [
                b"vllm",
                b"bench",
                b"serve",
                b"--num-warmups",
                b"16",
                b"--disable-tqdm",
                b"--header",
                b"x=a b",
                b"--header",
                b"y=c",
                b"--metadata",
                b'{"suite":"nightly"}',
                b"--zero",
                b"0",
            ],
        )


if __name__ == "__main__":
    unittest.main()
