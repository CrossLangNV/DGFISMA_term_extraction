"""
Estimate the speed of similar term extraction
"""

import time

import pandas as pd

from examples.similar_terms_main import get_corpus_example
from similar_terms.methods import SimilarWordsRetriever


def eval_load_speed():
    d = []
    for n in [0, 1, 10, 100, 1000, 10000, 100000, 1000000]:
        corpus = get_corpus_example(n)

        t0 = time.time()
        similar_words_retriever = SimilarWordsRetriever(corpus)
        t1 = time.time()

        d.append({'time': t1 - t0,
                  'n': len(corpus),
                  'info': 'loading object and preprocessing data'})

    df = pd.DataFrame(d)

    print(df)

    return df


def eval_retrieval_speed():
    corpus_voc = get_corpus_example(1000)
    similar_words_retriever = SimilarWordsRetriever(corpus_voc)

    corpus_all = get_corpus_example()
    d = []

    k = 10
    for n in [0, 1, 10, 100, 1000, 10000, 100000]:

        corpus_terms = corpus_all[max(0, len(corpus_all) - n):]

        t0 = time.time()

        for word in corpus_terms:
            a = similar_words_retriever.get_similar_k(word, k=k)

        t1 = time.time()

        d.append({'time': t1 - t0,
                  'n': len(corpus_terms),
                  'info': 'getting k similar words'})

    df = pd.DataFrame(d)

    print(df)

    return df


def eval_k_nearest_speed():
    corpus_voc = get_corpus_example(100000)
    similar_words_retriever = SimilarWordsRetriever(corpus_voc)

    corpus_all = get_corpus_example()

    corpus_terms = corpus_all[-100:]

    d = []
    for k in [1, 10, 100, 1000, 10000, 100000]:
        if k > len(corpus_voc):
            continue

        t0 = time.time()
        for word in corpus_terms:
            a = similar_words_retriever.get_similar_k(word, k=k)
        t1 = time.time()

        d.append({'time': t1 - t0,
                  'k': k,
                  'info': 'getting k similar words. range k'})

    df = pd.DataFrame(d)

    print(df)

    return df


if __name__ == '__main__':
    eval_load_speed()
    eval_retrieval_speed()
    eval_k_nearest_speed()
