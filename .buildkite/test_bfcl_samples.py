#!/usr/bin/env python3
"""Unit tests for BFCL → lm_eval sample conversion in run_bfcl.py."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN_BFCL = REPO / "lib" / "run_bfcl.py"


def load_run_bfcl():
    spec = importlib.util.spec_from_file_location("run_bfcl", RUN_BFCL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["run_bfcl"] = module
    spec.loader.exec_module(module)
    return module


run_bfcl = load_run_bfcl()


AST_FAILURE = {
    "id": "simple_python_13",
    "valid": False,
    "error_type": "type_error:nested",
    "error": ["Nested type checking failed for parameter 'interval'."],
    "prompt": {
        "id": "simple_python_13",
        "question": [
            [
                {
                    "role": "user",
                    "content": "Calculate the area under the curve y=x^2 from x=1 to x=3.",
                }
            ]
        ],
        "function": [
            {
                "name": "calculate_area_under_curve",
                "description": "Calculate the area under a mathematical function.",
                "parameters": {"type": "dict", "properties": {}, "required": []},
            }
        ],
    },
    "model_result_raw": [
        {
            "calculate_area_under_curve": (
                '{"function": "lambda x: x**2", "interval": [1, 3]}'
            )
        }
    ],
    "possible_answer": [
        {
            "calculate_area_under_curve": {
                "function": ["x**2"],
                "interval": [[1.0, 3.0]],
            }
        }
    ],
}

MULTI_TURN_FAILURE = {
    "id": "multi_turn_base_0",
    "valid": False,
    "error": {
        "error_type": "multi_turn:inference_failure",
        "error_message": "Model did not return a valid function call.",
    },
    "prompt": {
        "question": [[{"role": "user", "content": "Book a flight to NYC."}]],
        "function": [{"name": "book_flight", "parameters": {}}],
    },
    "model_result": [{"book_flight": '{"destination": "NYC"}'}],
    "possible_answer": [{"book_flight": {"destination": ["NYC"]}}],
}

AGENTIC_FAILURE = {
    "id": "web_search_0",
    "valid": False,
    "error_type": "checker_fail",
    "prompt": [{"role": "user", "content": "What is the weather in Paris?"}],
    "model_result": "It is sunny.",
    "possible_answer": ["sunny"],
}

RELEVANCE_FAILURE = {
    "id": "live_relevance_0",
    "valid": False,
    "error_type": "relevance",
    "prompt": {
        "question": [[{"role": "user", "content": "Tell me a joke."}]],
        "function": [],
    },
    "model_result": {"name": "irrelevant_tool", "arguments": "{}"},
}


class BfclSamplesTest(unittest.TestCase):
    def test_ast_failure_to_sample(self):
        sample = run_bfcl.bfcl_failure_to_sample(AST_FAILURE)
        self.assertIsInstance(sample["doc_id"], int)
        self.assertEqual(sample["doc_id"], run_bfcl._numeric_doc_id("simple_python_13"))
        self.assertEqual(sample["doc"]["answer"], "simple_python_13")
        self.assertIn("y=x^2", sample["doc"]["question"])
        self.assertEqual(sample["doc"]["functions"], ["calculate_area_under_curve"])
        self.assertEqual(sample["exact_match"], 0.0)
        self.assertEqual(sample["filtered_resps"], ["type_error:nested"])

    def test_multi_turn_failure_error_dict(self):
        sample = run_bfcl.bfcl_failure_to_sample(MULTI_TURN_FAILURE)
        self.assertIn("multi_turn:inference_failure", sample["filtered_resps"][0])
        self.assertIn("book_flight", sample["resps"][0][0])
        self.assertNotEqual(sample["resps"][0][0], "null")

    def test_agentic_failure_prompt_list(self):
        sample = run_bfcl.bfcl_failure_to_sample(AGENTIC_FAILURE)
        self.assertIn("Paris", sample["doc"]["question"])
        self.assertIn("sunny", sample["resps"][0][0])

    def test_relevance_failure_model_result(self):
        sample = run_bfcl.bfcl_failure_to_sample(RELEVANCE_FAILURE)
        self.assertIn("irrelevant_tool", sample["resps"][0][0])
        self.assertEqual(sample["filtered_resps"], ["relevance"])

    def test_success_to_sample(self):
        entry = {
            "id": "simple_python_0",
            "question": [
                [
                    {
                        "role": "user",
                        "content": "Find the area of a triangle with base 10 and height 5.",
                    }
                ]
            ],
            "function": [{"name": "calculate_triangle_area", "parameters": {}}],
            "ground_truth": [{"calculate_triangle_area": {"base": 10, "height": 5}}],
        }
        result_row = {
            "id": "simple_python_0",
            "result": [{"calculate_triangle_area": '{"base": 10, "height": 5}'}],
        }
        sample = run_bfcl.bfcl_success_to_sample(entry, result_row)
        self.assertEqual(sample["exact_match"], 1.0)
        self.assertIn("triangle", sample["doc"]["question"])

    def test_parse_score_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "score.json"
            path.write_text(
                json.dumps({"accuracy": 0.5, "correct_count": 1, "total_count": 2})
                + "\n"
                + json.dumps(AST_FAILURE)
                + "\n"
            )
            aggregate, failures = run_bfcl._parse_score_jsonl(path)
            self.assertEqual(aggregate["accuracy"], 0.5)
            self.assertIn("simple_python_13", failures)

    def test_find_category_artifact_exact_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            result_root = work / "result" / "Model" / "non_live"
            result_root.mkdir(parents=True)
            (result_root / "BFCL_v4_multiple_result.json").write_text("{}\n")
            (result_root / "BFCL_v4_live_multiple_result.json").write_text("{}\n")
            found = run_bfcl._find_category_artifact(work, "multiple", "result")
            self.assertEqual(found.name, "BFCL_v4_multiple_result.json")


if __name__ == "__main__":
    unittest.main()
