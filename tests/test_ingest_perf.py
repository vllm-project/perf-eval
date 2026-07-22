import argparse
import unittest

from lib import ingest_perf


class IngestPerfTopologyTests(unittest.TestCase):
    def test_disagg_uses_total_gpu_count_for_per_gpu_throughput(self):
        raw = {
            "total_token_throughput": 4000,
            "output_throughput": 1200,
            "max_concurrency": 3072,
        }
        args = argparse.Namespace(
            tp=16,
            gpu_count=40,
            disagg=True,
            is_multinode=True,
            date="2026-07-14 00:00:00",
            device="gb300",
            conc=3072,
            image="vllm:test",
            model="/model",
            precision="fp4",
            isl=8192,
            osl=1024,
        )

        transformed = ingest_perf.transform(raw, args)

        self.assertEqual(transformed["tp"], 16)
        self.assertEqual(transformed["disagg"], "true")
        self.assertEqual(transformed["is_multinode"], "true")
        self.assertEqual(transformed["tput_per_gpu"], 100)
        self.assertEqual(transformed["output_tput_per_gpu"], 30)
        self.assertEqual(transformed["input_tput_per_gpu"], 70)


if __name__ == "__main__":
    unittest.main()
