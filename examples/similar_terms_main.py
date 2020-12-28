import os
from typing import List

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), '..')


class Terms(list):
    def __init__(self, l):
        super(Terms, self).__init__(l)
        for e in self:
            assert isinstance(e, str)


def get_corpus_example(k: int = None) -> List[str]:
    """Get an example of a corpus containing terms.
    All terms are lower case!

    Args:
        k (int): optional, only read the first k terms for faster testing

    Returns:
        List of terms
    """

    path_words = os.path.join(ROOT, 'media/v1_dgf-voc.csv')

    l = pd.read_csv(path_words, usecols=[1], header=0)
    terms = Terms(l[l.keys()[0]])

    if k is not None:
        return terms[:k]

    return terms
