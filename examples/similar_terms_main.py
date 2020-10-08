import os
import time
from typing import List

import fasttext
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), '..')


def cosine_distances(X: np.ndarray, Y: np.ndarray):
    """ Implement own cosine distances to make it faster!

    Args:
        X:
        Y:

    Returns:

    """

    X = np.asarray(X)

    Y = np.asarray(Y)

    n_x, n_emb_x = X.shape
    n_y, n_emb_y = Y.shape

    assert n_emb_x == n_emb_y, 'embeddings do not match!'

    # X_norm = X / np.linalg.norm(X, axis=-1, keepdims=True)
    # # Y_norm = Y / np.linalg.norm(Y, axis=-1, keepdims=True)
    # Is already normalised!
    X_norm = X
    Y_norm = Y

    sim = np.dot(X_norm, Y_norm.T)

    return 1 - sim


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


def cleanup_corpus(corpus):
    # remove duplicates
    corpus_cleaned = list(dict.fromkeys(corpus))

    # remove if only contains underscore
    corpus_cleaned = [w for w in corpus_cleaned if w.strip('_')]

    return corpus_cleaned


def argpartitionsort(x: list, k: int):
    """Fast implementation of both sorting and partitioning

    Args:
        x: iterable
        k: number of elements to keep

    Returns:
        sorted indices of the k smallest elements of x
    """

    # WARNING! While does a lot of redundant work, np.argpartition does not sort
    idx_k_smallest = np.argpartition(x, k)[:k]

    # idx_sorted_k = np.argsort(d[idx_k_smallest])
    idx_sorted_k = sorted(idx_k_smallest, key=lambda i: x[i])

    return idx_sorted_k


def preprocessing(word):
    """

    Args:
        word:

    Returns:

    """
    # replace (space) with '_'

    word_pre = word[:]  # copy

    word_pre = word_pre.strip()
    word_pre = word_pre.lower()
    word_pre = word_pre.replace(' ', '_')
    return word_pre


class SimilarWords:
    def __init__(self,
                 corpus: List[str]):
        # TODO clean up corpus

        corpus = map(preprocessing, corpus)
        self.corpus = cleanup_corpus(corpus)  # remove duplicates

        #   TODO load from somewhere else
        # Fast text uses continues bag of words (CBOW) which means that for multigrams the average embedding is taken of each separate monogram.
        ROOT = os.path.join(os.path.dirname(__file__), '..')
        ftModel = fasttext.load_model(os.path.join(ROOT, 'media/dgf-model.tok.bin'))

        self.get_embedding = lambda term: ftModel.get_sentence_vector(term)

        self.embedding = np.array([self.get_embedding(term) for term in self.corpus])

    def get_similar_terms(self, term: str,
                          k=1):
        """

        Args:
            term (str):
            k (int): k most similar values

        Returns:
            a dictionary with index of terms and the terms
        """

        # TODO fasttext_model.get_nearest_neighbors(word, k) is much faster, BUT we need to be able to change it's dictionary.

        if isinstance(term, list):
            return {term_i: self.get_similar_terms(term_i, k=k) for term_i in term}

        assert isinstance(term, str)
        assert isinstance(k, int)

        term = preprocessing(term)

        emb_term = self.get_embedding(term)

        # Distance metric, so the lower, the more similar
        # TODO implement k nearest neighbours? Would this be a speedup? Could at least lead to less memory needed

        d = cosine_distances([emb_term], self.embedding)[0, ...]

        idx_sorted_k = argpartitionsort(d, k)

        terms = [self.corpus[i] for i in idx_sorted_k]
        distances = d[idx_sorted_k]

        return terms, idx_sorted_k, distances


def match_vocs(voc_a: List[str], voc_b: List[str], k: int,
               dtype=np.float32  # checking if this speeds things up
               ):
    """

    Args:
        voc_a:
        voc_b:

    Returns:
        a generator?
    """
    # TODO documentation

    # TODO flag to ignore exact match

    # TODO cava in class, save the dictionary, make it updatable? (might have to make sure that indices stay consistent)
    t0 = time.time()

    # Preprocess vocabularies

    voc_a = map(preprocessing, voc_a)
    voc_a = cleanup_corpus(voc_a)  # remove duplicates

    voc_b = map(preprocessing, voc_b)
    voc_b = cleanup_corpus(voc_b)  # remove duplicates

    t1 = time.time()

    # Load model
    ROOT = os.path.join(os.path.dirname(__file__), '..')
    ftModel = fasttext.load_model(os.path.join(ROOT, 'media/dgf-model.tok.bin'))

    get_embedding = lambda term: ftModel.get_sentence_vector(term)

    # generate embeddings

    t2 = time.time()

    embedding_a = np.array(list(map(get_embedding, voc_a)), dtype=dtype)
    embedding_b = np.array(list(map(get_embedding, voc_b)), dtype=dtype)
    # Make sure it's normalized (99% sure fasttext already does this, but better save than sorry
    embedding_a = embedding_a / np.linalg.norm(embedding_a, axis=-1, keepdims=True)
    embedding_b = embedding_b / np.linalg.norm(embedding_b, axis=-1, keepdims=True)

    # Calculate similarities
    t3 = time.time()
    # Get best k matches
    t4 = time.time()

    for i_a, term_a in enumerate(voc_a):
        # generator

        sim_i = np.dot(embedding_a[i_a, :], embedding_b.T)
        d_i = 1. - sim_i

        idx_sorted_k = argpartitionsort(d_i, k=k)

        terms_b = [voc_b[i] for i in idx_sorted_k]
        distances = d_i[idx_sorted_k]

        yield (term_a, (terms_b, idx_sorted_k, distances))

    t5 = time.time()

    print(f'Total time T = {t5 - t0} s')
    print(f'\tPreprocessing vocabularies T = {t1 - t0} s')
    print(f'\tLoading model T = {t2 - t1} s')
    print(f'\tgenerating embeddings T = {t3 - t2} s')
    print(f'\tCalculate similarities T = {t4 - t3} s')
    print(f'\tGet best k matches T = {t5 - t4} s')


if __name__ == '__main__':
    corpus = get_corpus_example(
        # 1000
    )

    n = 1000
    with open(os.path.join(ROOT, 'media/matches_output_example.csv'), 'w+') as f:
        for a in match_vocs(corpus[:n], corpus, k=5):
            term, (sim_terms, indices, distances) = a
            s = f'{term}; [{", ".join(sim_terms)}]; [{", ".join(map(str, indices))}]; [{", ".join(map(str, distances))}]'
            f.write(s + '\n')

    print('Stop here')

    # similar_words = SimilarWords(corpus=corpus)
    #
    # n_iter = 10
    # start = time.time()
    #
    # a = similar_words.get_similar_terms(corpus[:n_iter],
    #                                     k=5)
    #
    # finish = time.time()
    #
    # print(f'give it already a list (n={n_iter}) of terms took {finish-start} s.')
    #
    # start = time.time()
    #
    # for i in range(n_iter):
    #     a = similar_words.get_similar_terms(corpus[i],
    #                                         k=5)
    #
    # finish = time.time()
    #
    # print(f'Delta T = {(finish - start) / n_iter} per get_similar')
    # print(f'T estimate for whole dataset: {(finish - start) / n_iter * len(corpus)}')
    #
    # print(a)
    # print('Done')
