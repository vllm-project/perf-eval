from lm_eval.tasks.aime.utils import process_results  # noqa: F401


def process_docs(dataset, n_repeats=8):
    """Each problem n_repeats times; mean exact_match over them is mean@n."""
    import datasets

    return datasets.concatenate_datasets([dataset] * n_repeats)
