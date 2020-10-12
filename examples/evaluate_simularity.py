"""
Based on a list of pairs of words, expecting that the embedding size is identical,
maximise their simularity
"""

import itertools
import os

import numpy as np
import pandas as pd

from similar_terms.methods import SimilarWordsRetriever, get_similarity_fast
from similar_terms.preprocessing import *
from similar_terms.preprocessing import preprocessing_word


def get_pairs():
    """TODO make much more extensive

    Returns:

    """
    l = [('EU', 'European Union'),
         ('energy', 'nuclear power')]

    return l


def get_pairs_abbreviation():
    """Based on list of abbreviation, we can use it to see how well the embedding is to link abbreviations (if it even can link them)

    Returns: list of strings
    """

    root = os.path.join(os.path.dirname(__file__), '..')
    eurlex_abbv = os.path.join(root, 'media/eurlex-abbv.csv')

    df = pd.read_csv(eurlex_abbv)

    l = list(zip(df['abbv'], df['term']))

    # Seems to be some NaN's contained in df. only keep with
    l = [(a, b) for a, b in l if isinstance(a, str) and isinstance(b, str)]

    return l


def eval_preprocessing():
    """ Evaluate different types of preprocessing steps.

    Returns:
        None
    """

    sim_words_retriever = SimilarWordsRetriever([])

    if 0:
        l = get_pairs()
    elif 1:
        l = get_pairs_abbreviation()

    def get_emb(s):
        emb = sim_words_retriever.get_embedding(s)
        return emb

    preprocessers = {'do nothing': preproc1,
                     'reverse order': preproc2,
                     'preprocessing + lowercase': lambda s: preprocessing_word(s, True),
                     'preprocessing': lambda s: preprocessing_word(s, False)}

    f_lst = (
        lower_case,
        strip,
        single_space,
        join_words,
        single_underscore
    )

    def preproc_generator():

        p = itertools.product([False, True], repeat=len(f_lst))

        l_f = {}

        def f_joined(s, b_lst):

            for b_j, f_j in zip(b_lst, f_lst):
                if b_j:
                    s = f_j(s)

            return s

        for p_i in p:
            # We already have to give p_i!!
            lam = lambda s, p_i=p_i: f_joined(s, p_i[:])
            l_f['_'.join(map(str, map(int, p_i)))] = lam

        return l_f

    preprocessers.update(preproc_generator())

    foo = {}
    for name, preprocesser in preprocessers.items():

        score = []

        for s1, s2 in l:
            s1, s2 = map(preprocesser, (s1, s2))

            e1 = get_emb(s1)
            e2 = get_emb(s2)

            score_i = get_similarity_fast(e1, e2)
            score.append(score_i)

        foo[name] = np.mean(score)

    foo_sorted = {k: v for k, v in sorted(foo.items(), key=lambda item: item[1], reverse=True)}
    print('sorted preprocessing functions:')
    print(foo_sorted)

    def check_preproc_generator_key(k):

        try:
            i_lst = list(map(int, k.split('_')))  # already cast to list to check if actually integers
            return True
        except:
            return

    max_v = max(foo_sorted.values())
    keys_max = [k for k, v in foo_sorted.items() if v == max_v and check_preproc_generator_key(k)]

    # Check which flags are important!

    def check_equal(lst: list):
        """check if all values in list are equal"""
        return lst[1:] == lst[:-1]

    for i, b_i_lst in enumerate(zip(*map(lambda s: s.split('_'), keys_max))):

        b_i_lst = list(map(int, b_i_lst))

        if check_equal(b_i_lst):
            b_i = b_i_lst[0]

            print(f'set to {b_i}: {f_lst[i]}')

    return


def preproc1(s):
    """ Do nothing.

    Args:
        s:

    Returns:

    """
    return s


def preproc2(s):
    """ Opposite order. Should be worse

    Args:
        s:

    Returns:

    """
    return s[::-1]


if __name__ == '__main__':
    eval_preprocessing()
