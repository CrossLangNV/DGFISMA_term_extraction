"""
All methods that are required for similar term retrieval.
Might be moved to separate files in the future.
"""

import os
from typing import List

import fasttext
import numpy as np

from similar_terms.preprocessing import preprocessing_word


class Vocabulary(list):
    def __init__(self, l: List[str]):
        super(Vocabulary, self).__init__(map(str, l))


class SimilarWordsRetriever:
    """ Based on a vocubaluray, can find similar words.
    Similar words are found by sorting based on the cosine similarity of word embeddings.
    """

    def __init__(self, vocabulary: List[str]):
        vocabulary = Vocabulary(vocabulary)

        self._vocabulary = vocabulary

        self._map_cleaned_orig = {word: i for i, word in enumerate(map(preprocessing_word, vocabulary))}

        self._voc_cleaned = list(self._map_cleaned_orig)

        # TODO choose model
        ROOT = os.path.join(os.path.dirname(__file__), '..')
        self._ftModel = fasttext.load_model(os.path.join(ROOT, 'media/dgf-model.tok.bin'))

        self._embedding_voc_cleaned = self.get_embedding(self._voc_cleaned)

    def get_embedding(self, s: (str, List[str])) -> np.ndarray:
        """  Retrieve the embedded vector representation of a string or list of strings.
        Used for similar word retrieval

        Args:
            s: string or list of strings. If Multigram, processed as continues bag of words (CBOW).
                if you don't want to use CBOW, replace spaces with underscores.

        Returns:
            1D (or 2D if input is list) embedding vector.
        """

        if isinstance(s, str):
            emb = self._ftModel.get_sentence_vector(s)

        else:
            emb = np.asarray([self._ftModel.get_sentence_vector(s_i) for s_i in s])

        # Manually normalising, which is necessary in case of CBOW (average of normalised vectors is not normed)
        return emb / np.linalg.norm(emb, axis=-1, keepdims=True)

    def get_voc(self):
        return self._vocabulary

    def get_original_word(self, word_pre_processed: str):
        i = self._map_cleaned_orig.get(word_pre_processed, None)
        if i is not None:
            return self.get_voc()[i]

    def get_similar_foo(self, term, term_voc):
        # TODO name

        from examples.similar_terms_main import match_vocs

        return list(match_vocs([term], term_voc, k=5))[0]

    def get_similar_k(self, term, k=1):
        # TODO flag to exclude self!, default on
        # TODO exclude based on preprocessing or we could just exclude sim of 1

        sim = self._get_sim(term)
        idx_sorted_k = argpartitionsort(sim, k, reverse=True)

        terms_orig = [self.get_original_word(self._voc_cleaned[i]) for i in idx_sorted_k]

        similarities = sim[idx_sorted_k]

        return SimilarTerms(terms_orig, similarities)

    def get_similar_thresh(self, term, thresh=.5):
        # TODO flag to excluse self, default on

        sim = self._get_sim(term)

        idx_sorted_k = sorted(np.argwhere(sim >= thresh).reshape((-1,)), reverse=True, key=sim.take)

        terms_orig = [self.get_original_word(self._voc_cleaned[i]) for i in idx_sorted_k]
        similarities = sim[idx_sorted_k]

        return SimilarTerms(terms_orig, similarities)

    def _get_sim(self, term):
        term_pre = preprocessing_word(term)

        # return as a (1 x n) matrix
        emb_term = self.get_embedding([term_pre])

        sim = get_similarity_fast(emb_term, self._embedding_voc_cleaned)[0]
        return sim


class SimilarTerms(dict):
    def __init__(self, terms_orig, similarities):
        self.update({'original terms': terms_orig,
                     'score': similarities})


def argpartitionsort(x: list, k: int, reverse: bool = False):
    """Fast implementation of both sorting and partitioning

    Args:
        x: iterable
        k: number of elements to keep, should be bigger than 0
        reverse: if True, sort from big to small

    Returns:
        sorted indices of the k smallest elements of x
    """

    if k <= 0:
        raise ValueError(f'k should be positive: {k}')

    # return everything
    if k >= len(x):

        idx_k_subset = np.arange(len(x))

    elif reverse:
        # WARNING! While does a lot of redundant work, np.argpartition does not sort
        idx_k_subset = np.argpartition(x, -k, )[-k:]
    else:
        # WARNING! While does a lot of redundant work, np.argpartition does not sort
        idx_k_subset = np.argpartition(x, k, )[:k]

    idx_sorted_k = sorted(idx_k_subset, reverse=reverse, key=lambda i: x[i])

    return idx_sorted_k


def get_similarity_fast(emb1: np.ndarray, emb2: np.ndarray):
    """ Fast implementation of cosine similarity. Expects that both embeddings are normalised.
    As this is called so many times, to improve speed, no checks are done on input!

    Args:
        emb1: embedding matrix of shape (n1, n_emb)
        emb2: embedding matrix of shape (n2, n_emb)

    Returns: similarity matrix of shape (n1, n2) with values range [-1, 1]

    """

    sim = np.dot(emb1, emb2.T)
    return sim
