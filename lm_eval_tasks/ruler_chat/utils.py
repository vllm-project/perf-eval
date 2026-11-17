"""lm-eval's 8 RULER NIAH tasks, rebuilt with lm-eval's own generator at NUM_SAMPLES a length.

WORKAROUND: lm-eval's NIAH generator hard-codes num_samples=500, and --limit only trims after
all 500 are built. Remove the dataset wrappers once lm-eval reads num_samples from metadata:
https://github.com/EleutherAI/lm-evaluation-harness/pull/4640
"""

import random

import datasets
import numpy as np
from lm_eval.tasks.ruler import niah_utils
from lm_eval.tasks.ruler.common_utils import (
    get_tokenizer,
    postprocess_pred,
    resolve_tokenizer_name,
    string_match_all,
)
from lm_eval.tasks.ruler.prepare_niah import generate_samples, get_haystack

NUM_SAMPLES = 40  # moves to task metadata as num_samples once https://github.com/EleutherAI/lm-evaluation-harness/pull/4640 lands
SEED = 42  # RULER's default random_seed


# Workaround: remove once https://github.com/EleutherAI/lm-evaluation-harness/pull/4640 lands.
def _niah(type_haystack, type_needle_k, type_needle_v, **num_needles):
    """custom_dataset that builds NUM_SAMPLES docs for each length in metadata."""

    def dataset(**kwargs):
        tokenizer = get_tokenizer(resolve_tokenizer_name(kwargs))
        docs = []
        for seq in kwargs["max_seq_lengths"]:
            # Seed per length so a length's samples don't depend on what was built before it.
            random.seed(SEED)
            np.random.seed(SEED)
            docs += generate_samples(
                get_haystack(type_haystack=type_haystack),
                max_seq_length=seq,
                template=niah_utils.TEMPLATE,
                type_haystack=type_haystack,
                type_needle_k=type_needle_k,
                type_needle_v=type_needle_v,
                num_samples=NUM_SAMPLES,
                TOKENIZER=tokenizer,
                **num_needles,
            )
        return {"test": datasets.Dataset.from_list(docs, split=datasets.Split.TEST)}

    return dataset


# Same haystack/needle settings as lm-eval's niah_utils.
niah_single_1 = _niah("repeat", "words", "numbers")
niah_single_2 = _niah("essay", "words", "numbers")
niah_single_3 = _niah("essay", "words", "uuids")
niah_multikey_1 = _niah("essay", "words", "numbers", num_needle_k=4)
niah_multikey_2 = _niah("needle", "words", "numbers")
niah_multikey_3 = _niah("needle", "uuids", "uuids")
niah_multivalue = _niah("essay", "words", "numbers", num_needle_v=4)
niah_multiquery = _niah("essay", "words", "numbers", num_needle_q=4)


def process_results(doc, results):
    """Score each doc under its own length only: the fraction of its answers in the response."""
    return {str(doc["max_length"]): string_match_all(postprocess_pred(results), [doc["outputs"]])}
