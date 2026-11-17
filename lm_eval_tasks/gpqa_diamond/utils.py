# process_docs is verbatim from InferenceX (Apache-2.0, https://github.com/SemiAnalysisAI/InferenceX/blob/99d0296a292dabacb817b8e73f74a730f13642ad/LICENSE):
# https://github.com/SemiAnalysisAI/InferenceX/blob/99d0296a292dabacb817b8e73f74a730f13642ad/inferencex-e2e/infx/evals/utils.py
import random
import re

import datasets

LETTERS = "ABCD"


def process_docs(
    dataset: datasets.Dataset, n_repeats: int = 2, seed: int = 3407
) -> datasets.Dataset:
    """Return dataset repeated n_repeats times with answers shuffled into A to D by seed.

    Each row gets its correct letter in answer and its repeat index in repeat_id.
    """
    rng = random.Random(seed)  # noqa: S311
    docs = list(dataset)

    rows = []
    for r in range(n_repeats):
        for doc in docs:
            base_choices = [
                doc["Correct Answer"],
                doc["Incorrect Answer 1"],
                doc["Incorrect Answer 2"],
                doc["Incorrect Answer 3"],
            ]
            perm = rng.sample(range(4), 4)

            new_choices = [base_choices[j] for j in perm]
            correct_letter = LETTERS[perm.index(0)]

            new_doc = dict(doc)
            new_doc["A"], new_doc["B"], new_doc["C"], new_doc["D"] = new_choices
            new_doc["answer"] = correct_letter
            new_doc["repeat_id"] = r
            rows.append(new_doc)

    return datasets.Dataset.from_list(rows)


# parse_sampled_answer is verbatim from the GPQA authors' AnswerPredictor (MIT, Copyright (c) 2022 I. David Rein,
# https://github.com/idavidrein/gpqa/blob/56686c06f5e19865c153de0fdb11be3890014df7/LICENSE):
# https://github.com/idavidrein/gpqa/blob/56686c06f5e19865c153de0fdb11be3890014df7/baselines/run_baseline.py
LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}


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
