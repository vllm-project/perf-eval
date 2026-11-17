"""lm-eval's 13 RULER tasks, rebuilt with lm-eval's own generators at NUM_SAMPLES a length.

WORKAROUND: every lm-eval RULER generator hard-codes num_samples=500, and --limit only trims after
all 500 are built. Remove the dataset wrappers once lm-eval reads num_samples from metadata.
"""

import random
import string

import datasets
import numpy as np
from lm_eval.tasks.ruler import cwe_utils, fwe_utils, niah_utils, qa_utils, vt_utils
from lm_eval.tasks.ruler.common_utils import (
    get_tokenizer,
    postprocess_pred,
    resolve_tokenizer_name,
    string_match_all,
)
from lm_eval.tasks.ruler.prepare_niah import generate_samples, get_haystack

NUM_SAMPLES = 40
SEED = 42  # RULER's default random_seed


def _dataset(build):
    """custom_dataset that runs build(tokenizer, max_seq_length) for each length in metadata."""

    def dataset(**kwargs):
        tokenizer = get_tokenizer(resolve_tokenizer_name(kwargs))
        docs = []
        for seq in kwargs["max_seq_lengths"]:
            # Seed per length so a length's samples don't depend on what was built before it.
            random.seed(SEED)
            np.random.seed(SEED)
            cwe_utils.RNG.seed(SEED)
            docs += build(tokenizer, seq)
        return {"test": datasets.Dataset.from_list(docs, split=datasets.Split.TEST)}

    return dataset


def _niah(type_haystack, type_needle_k, type_needle_v, **num_needles):
    def build(tokenizer, seq):
        return generate_samples(
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

    return _dataset(build)


# Same haystack/needle settings as lm-eval's niah_utils.
niah_single_1 = _niah("repeat", "words", "numbers")
niah_single_2 = _niah("essay", "words", "numbers")
niah_single_3 = _niah("essay", "words", "uuids")
niah_multikey_1 = _niah("essay", "words", "numbers", num_needle_k=4)
niah_multikey_2 = _niah("needle", "words", "numbers")
niah_multikey_3 = _niah("needle", "uuids", "uuids")
niah_multivalue = _niah("essay", "words", "numbers", num_needle_v=4)
niah_multiquery = _niah("essay", "words", "numbers", num_needle_q=4)


def _generate_chains(num_chains, num_hops, is_icl=False):
    """WORKAROUND: lm-eval's vt_utils.generate_chains with its name-collision bug fixed.

    When two drawn names collide, the redraw is appended after the duplicate, so the list splits
    into a short extra chain and indexing it raises IndexError. Remove once the pinned lm-eval
    dedupes vars_all.
    """
    k = 5 if not is_icl else 3
    num_hops = num_hops if not is_icl else min(10, num_hops)
    vars_all = [
        "".join(random.choices(string.ascii_uppercase, k=k)).upper()
        for _ in range((num_hops + 1) * num_chains)
    ]
    while len(set(vars_all)) < num_chains * (num_hops + 1):
        vars_all.append("".join(random.choices(string.ascii_uppercase, k=k)).upper())
    vars_all = list(dict.fromkeys(vars_all))  # the fix

    vars_ret = []
    chains_ret = []
    for i in range(0, len(vars_all), num_hops + 1):
        this_vars = vars_all[i : i + num_hops + 1]
        vars_ret.append(this_vars)
        this_chain = [f"VAR {this_vars[0]} = {np.random.randint(10000, 99999)}"]
        for j in range(num_hops):
            this_chain.append(f"VAR {this_vars[j + 1]} = VAR {this_vars[j]} ")
        chains_ret.append(this_chain)
    return vars_ret, chains_ret


vt_utils.generate_chains = _generate_chains


def _vt(tokenizer, seq):
    icl_example = vt_utils.sys_vartrack_w_noise_random(
        tokenizer=tokenizer, num_samples=1, max_seq_length=500, incremental=5
    )[0]
    return vt_utils.sys_vartrack_w_noise_random(
        tokenizer=tokenizer, num_samples=NUM_SAMPLES, max_seq_length=seq, icl_example=icl_example
    )


def _cwe(tokenizer, seq):
    return cwe_utils.sys_word_pair_random(
        num_samples=NUM_SAMPLES, max_seq_length=seq, tokenizer=tokenizer
    )


def _fwe(tokenizer, seq):
    return fwe_utils.sys_kwext(tokenizer=tokenizer, max_seq_length=seq, num_samples=NUM_SAMPLES)


def _qa(read):
    def build(tokenizer, seq):
        qas, docs = read()
        return qa_utils.generate_samples(
            tokenizer, docs, qas, max_seq_length=seq, num_samples=NUM_SAMPLES, tokens_to_generate=32
        )

    return build


vt = _dataset(_vt)
cwe = _dataset(_cwe)
fwe = _dataset(_fwe)
qa_squad = _dataset(_qa(qa_utils.read_squad))
qa_hotpot = _dataset(_qa(qa_utils.read_hotpotqa))


def process_results(doc, results):
    """Score each doc under its own length only: the fraction of its answers in the response."""
    return {str(doc["max_length"]): string_match_all(postprocess_pred(results), [doc["outputs"]])}


def process_results_qa(doc, results):
    """WORKAROUND: RULER's QA score, 1 if the response contains any reference answer.

    lm-eval's string_match_part scores the fraction of references matched instead
    (lm-evaluation-harness#4272). Remove once the pinned lm-eval has that fix.
    """
    pred = postprocess_pred(results)[0].lower()
    return {str(doc["max_length"]): float(any(r.lower() in pred for r in doc["outputs"]))}
