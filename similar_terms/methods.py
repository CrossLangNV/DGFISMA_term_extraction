"""
All methods that are required for similar term retrieval.
Might be moved to separate files in the future.
"""

import os
from typing import List

import fasttext
import numpy as np

from media.data import get_filename_fasttext_model
from similar_terms.preprocessing import preprocessing_word

FILENAME_FASTTEXT_MODEL = os.environ.get('FASTTEXT_PATH', get_filename_fasttext_model())


class Vocabulary(list):
    def __init__(self, l: List[str]):
        super(Vocabulary, self).__init__(map(str, l))


class SimilarWordsRetriever:
    """ Based on a vocubaluray, can find similar words.
    Similar words are found by sorting based on the cosine similarity of word embeddings.
    """

    def __init__(self,
                 vocabulary: List[str] = [],
                 fasttext_model: fasttext.FastText = None,
                 filename_fasttext_model=FILENAME_FASTTEXT_MODEL,
                 preprocessor=preprocessing_word):
        """

        Args:
            vocabulary:
            fasttext_model:
            filename_fasttext_model: If fasttext_model is None, this should be non-empty
            preprocessor:
        """

        if fasttext_model is None:
            self._ftModel = fasttext.load_model(filename_fasttext_model)
        else:
            self._ftModel = fasttext_model

        self.preprocessor = preprocessor

        self.set_vocabulary(vocabulary)

    def set_vocabulary(self, vocabulary):
        """

        Args:
            vocabulary:

        Returns:

        """
        vocabulary = Vocabulary(vocabulary)

        self._vocabulary = vocabulary
        self._map_cleaned_orig = {word: i for i, word in enumerate(map(self.preprocessor, vocabulary))}
        self._voc_cleaned = list(self._map_cleaned_orig)
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
        emb_norm = np.linalg.norm(emb, axis=-1, keepdims=True)
        return np.divide(emb, emb_norm, out=np.zeros_like(emb), where=emb_norm != 0)

    def get_voc(self):
        return self._vocabulary

    def get_original_word(self, word_pre_processed: str):
        i = self._map_cleaned_orig.get(word_pre_processed, None)
        if i is not None:
            return self.get_voc()[i]

    def get_similar_k(self,
                      term,
                      k=1,
                      include_self=True):
        # TODO flag to exclude self!, default on
        # TODO exclude based on preprocessing or we could just exclude sim of 1

        sim = self._get_sim(term)

        # TODO remove self.
        # set sim == 0 to -1?
        if not include_self:
            sim[sim >= 1.] = -np.inf  # Make sure, it's lowest on list.

        idx_sorted_k = argpartitionsort(sim, k, reverse=True)

        return self._get_similar_from_idx(sim, idx_sorted_k)

    def get_similar_thresh(self, term, thresh=.5):
        # TODO flag to excluse self, default on

        sim = self._get_sim(term)

        idx_sorted_k = sorted(np.argwhere(sim >= thresh).reshape((-1,)), reverse=True, key=sim.take)

        return self._get_similar_from_idx(sim, idx_sorted_k)

    def _get_similar_from_idx(self, sim, idx):
        """ Shared private method

        Args:
            idx:

        Returns:

        """

        terms_orig = [self.get_original_word(self._voc_cleaned[i]) for i in idx]
        similarities = sim[idx]

        return SimilarTerms(terms_orig, similarities)

    def _get_sim(self, term):
        term_pre = self.preprocessor(term)

        # return as a (1 x n) matrix
        emb_term = self.get_embedding([term_pre])

        sim = get_similarity_fast(emb_term, self._embedding_voc_cleaned)[0]
        return sim

    def get_sim_between_words(self, term1: str, term2: str):

        term_pre1 = self.preprocessor(term1)
        term_pre2 = self.preprocessor(term2)

        # return as a (1 x n) matrix
        emb_term1 = self.get_embedding([term_pre1])
        emb_term2 = self.get_embedding([term_pre2])

        sim = float(get_similarity_fast(emb_term1, emb_term2))
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
