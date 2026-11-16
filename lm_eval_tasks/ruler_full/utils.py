"""WORKAROUND: lm-eval's ruler_vt crashes when two variable names collide; patch in the upstream fix.

Real fix: the same one-line change in lm-eval's lm_eval/tasks/ruler/vt_utils.py (and NVIDIA/RULER's
variable_tracking.py). Remove this file and ruler_vt's custom_dataset once the pinned lm-eval has it.
"""

import random
import string

import numpy as np
from lm_eval.tasks.ruler import vt_utils


def generate_chains(num_chains, num_hops, is_icl=False):
    k = 5 if not is_icl else 3
    num_hops = num_hops if not is_icl else min(10, num_hops)
    vars_all = [
        "".join(random.choices(string.ascii_uppercase, k=k)).upper()
        for _ in range((num_hops + 1) * num_chains)
    ]
    while len(set(vars_all)) < num_chains * (num_hops + 1):
        vars_all.append("".join(random.choices(string.ascii_uppercase, k=k)).upper())
    # The fix: redraws land after duplicates; drop them so chunks stay full.
    vars_all = list(dict.fromkeys(vars_all))

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


vt_utils.generate_chains = generate_chains
get_vt_dataset = vt_utils.get_vt_dataset
