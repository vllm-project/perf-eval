# The GPQA authors' zero-shot baseline (MIT, Copyright (c) 2022 I. David Rein,
# https://github.com/idavidrein/gpqa/blob/56686c06f5e19865c153de0fdb11be3890014df7/LICENSE):
# process_docs ports load_examples from baselines/utils.py (seed 0, the default in baselines/run_baseline.py);
# parse_sampled_answer is verbatim from baselines/run_baseline.py.
import random
import re

import datasets

LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}


def process_docs(dataset: datasets.Dataset, seed: int = 0) -> datasets.Dataset:
    # Own Random instead of random.seed(seed): the same permutations without reseeding lm-eval's global RNG.
    rng = random.Random(seed)  # noqa: S311

    def shuffle_choices(row):
        list_choices = [row['Incorrect Answer 1'], row['Incorrect Answer 2'], row['Incorrect Answer 3'], row['Correct Answer']]
        rng.shuffle(list_choices)
        row["A"], row["B"], row["C"], row["D"] = list_choices
        row["answer"] = "ABCD"[list_choices.index(row['Correct Answer'])]
        return row

    return datasets.Dataset.from_list([shuffle_choices(dict(row)) for row in dataset])


def parse_sampled_answer(answer):
    patterns = [r'answer is \((.)\)', r'Answer: \((.)\)', r'answer: \((.)\)', r'answer \((.)\)', r'\((.)\)']
    for pattern in patterns:
        match = re.search(pattern, answer)
        if match and match.group(1) in LETTER_TO_INDEX:
            return match.group(1)
    return None


def process_results(doc, results):
    # Unparsed answers count as wrong, as in the authors' baseline.
    return {"exact_match": float(parse_sampled_answer(results[0]) == doc["answer"])}
