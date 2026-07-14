import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class RunVllmBenchTests(unittest.TestCase):
    def test_waits_for_delayed_result_file_visibility(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_bin = temp_path / "bin"
            fake_bin.mkdir()
            fake_vllm = fake_bin / "vllm"
            fake_vllm.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    set -euo pipefail
                    result_file=""
                    while (($#)); do
                      if [[ "$1" == "--result-filename" ]]; then
                        result_file=$2
                        break
                      fi
                      shift
                    done
                    [[ -n "$result_file" ]]
                    [[ ! -e "$result_file" ]]
                    (
                      sleep 0.2
                      printf '{"completed": 1, "failed": 0}\n' >"$result_file"
                    ) >/dev/null 2>&1 &
                    """
                )
            )
            fake_vllm.chmod(0o755)
            results_path = temp_path / "results"
            results_path.mkdir()
            (results_path / "bench-delayed.json").write_text(
                '{"completed": 1, "failed": 0, "stale": true}\n'
            )

            command = textwrap.dedent(
                f"""\
                set -euo pipefail
                source {REPO_ROOT / 'lib' / 'run_vllm_bench.sh'}
                export WORKLOAD_SERVER_RUNTIME=native
                run_vllm_bench unused 8000 model delayed - random \\
                  32 8 1 1 - - false {results_path}
                """
            )
            environment = os.environ.copy()
            environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
            completed = subprocess.run(
                ["bash", "-c", command],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                capture_output=True,
                timeout=10,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("saved", completed.stdout)
            result = (results_path / "bench-delayed.json").read_text()
            self.assertNotIn("stale", result)


if __name__ == "__main__":
    unittest.main()
