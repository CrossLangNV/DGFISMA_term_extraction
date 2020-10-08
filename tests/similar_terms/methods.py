import unittest

import numpy as np

from examples.similar_terms_main import SimilarWords
from similar_terms.methods import SimilarWordsRetriever, preprocessing_word, argpartitionsort

VOC_RANDOM_CHAR = ["FyUVpwezcV", "5pDYh8e9aN", "inPxwYCAxx", "zFgAXeHetZ", "f3qn2vpSSE", "RpHCTNu5MU", "7K7kqt4ZJG", "MgphBv2ynj", "anyxprP7GS", "jdVUDJXRk5"]
VOC_RANDOM_WORDS = ["subway", "craft", "injection", "hen", "general", "us", "color-blind", "pool", "feminist", "basin", "polite", "combination"]
VOC_RANDOM_SENTENCE = ["They throw cabbage that turns your brain into emotional baggage.",
                       "He learned the important lesson that a picnic at the beach on a windy day is a bad idea.",
                       "He fumbled in the darkness looking for the light switch, but when he finally found it there was someone already there.",
                       "As he looked out the window, he saw a clown walk by."]
VOC_RANDOM_SYMBOLS = ["va,yS}WAL(", "CE*2Ag$n;J", "ELMN.dTMnR", "c%PL;5]UUn", ",HC=;MY)qU", "y*z;.[(gU!", "+czkW$F.Fe", "&Af[+ufx,{",
       "S:td}W5)9f", "ft)7T7(,/F"]
VOC_SIMILAR_WORDS = ["container ship",
                     "Container  ship",
                     "Container_ship ",
                     "containership",
                     " container_ship ",
                     " [container_ship] ",
                     " container ship.",
                     ]

VOC_ALL = VOC_RANDOM_CHAR + VOC_RANDOM_WORDS + VOC_RANDOM_SENTENCE + VOC_RANDOM_SYMBOLS+ VOC_SIMILAR_WORDS

class TestSimilarWordsRetriever(unittest.TestCase):

    def test_init(self):
        """
        Empty vocabulary should not crash although it's not helpful
        """

        similar_words_retriever = SimilarWordsRetriever([])

        self.assertIsNotNone(similar_words_retriever)

    def test_init_voc(self):
        similar_words_retriever = SimilarWordsRetriever(['test'])

        self.assertIsNotNone(similar_words_retriever)

    def test_init_dif_vocs(self):

        for voc in [VOC_RANDOM_CHAR, VOC_RANDOM_SENTENCE, VOC_RANDOM_WORDS, VOC_SIMILAR_WORDS, VOC_RANDOM_SYMBOLS]:
            with self.subTest():
                similar_words_retriever = SimilarWordsRetriever(voc)

                self.assertIsNotNone(similar_words_retriever)

    def test_get_voc(self):

        for voc in [VOC_RANDOM_CHAR, VOC_RANDOM_SENTENCE, VOC_RANDOM_WORDS, VOC_SIMILAR_WORDS, VOC_RANDOM_SYMBOLS]:
            with self.subTest():
                similar_words_retriever = SimilarWordsRetriever(voc)
                self.assertEqual(voc, similar_words_retriever.get_voc())

    def test_get_all(self):

        for voc in [VOC_RANDOM_CHAR, VOC_RANDOM_SENTENCE, VOC_RANDOM_WORDS, VOC_SIMILAR_WORDS, VOC_RANDOM_SYMBOLS]:
            with self.subTest():
                similar_words_retriever = SimilarWordsRetriever(voc)
                # Thresh of 0 should be the maximum distance of cosine with normalised vectors
                sim_words = similar_words_retriever.get_similar_thresh('test', 0)
                self.assertLess(0, len(sim_words['original terms']))

    def test_get_all_from_voc(self):
        """ what you get back should be included in the original provided voc.

        Returns:

        """
        for voc in [VOC_RANDOM_CHAR, VOC_RANDOM_SENTENCE, VOC_RANDOM_WORDS, VOC_SIMILAR_WORDS, VOC_RANDOM_SYMBOLS]:

            with self.subTest():

                similar_words_retriever = SimilarWordsRetriever(voc)
                for term in voc:
                    "Thresh of -1 should be the maximum distance of cosine"
                    sim_words = similar_words_retriever.get_similar_k(term, k=len(voc))

                    self.assertTrue(len(sim_words['original terms']) > 0, 'Should return multiple terms')
                    self.assertTrue(set(sim_words['original terms']).issubset(voc), 'Should return strings from vocubulary')

    def test_get_similar_k(self):

        for voc in [VOC_RANDOM_CHAR, VOC_RANDOM_SENTENCE, VOC_RANDOM_WORDS, VOC_SIMILAR_WORDS, VOC_RANDOM_SYMBOLS]:

            k = len(voc) // 2
            with self.subTest():
                similar_words_retriever = SimilarWordsRetriever(voc)
                for term in voc:

                    sim_words = similar_words_retriever.get_similar_k(term, k=k)
                    self.assertEqual(k, len(sim_words['original terms']))

    def test_get_similar_thresh(self):
        """
        Check if all returned thresholds in range?

        Returns:
            None
        """

        for voc in [VOC_RANDOM_CHAR, VOC_RANDOM_SENTENCE, VOC_RANDOM_WORDS, VOC_SIMILAR_WORDS, VOC_RANDOM_SYMBOLS]:

            thresh = .5

            with self.subTest():
                similar_words_retriever = SimilarWordsRetriever(voc)
                for term in voc:

                    sim_words = similar_words_retriever.get_similar_thresh(term, thresh=thresh)
                    scores = sim_words['score']
                    self.assertLess(thresh, scores[0])

        with self.subTest('Test thresholds'):
            similar_words_retriever = SimilarWordsRetriever(VOC_RANDOM_SENTENCE)

            for thresh in (-100, -35.18, -1, -.99, -0.135, 0, .51, .99, 1, 1., 18834):
                sim_words = similar_words_retriever.get_similar_thresh('test term', thresh=thresh)

                scores = sim_words['score']

                self.assertIsInstance(list(scores), list)

                if len(scores):
                    score = scores[0]
                    self.assertLess(thresh, score)

    def test_get_preprocessed(self):

        voc = VOC_SIMILAR_WORDS
        similar_words_retriever = SimilarWordsRetriever(voc)

        voc_cleaned = similar_words_retriever._map_cleaned_orig.keys()

        self.assertLess(len(voc_cleaned), len(voc), 'should be subset')

    def test_get_original_word(self):

        voc = VOC_SIMILAR_WORDS
        similar_words_retriever = SimilarWordsRetriever(voc)

        voc_cleaned = similar_words_retriever._map_cleaned_orig.keys()

        # with self.subTest():
        for word_cleaned in voc_cleaned:
            word_orig = similar_words_retriever.get_original_word(word_cleaned)

            self.assertTrue(word_orig in voc)


    def test_order(self):
        word = 'banks'
        voc = ['bank', 'money', 'agewgfsz']
        idx_sorted = [0, 1, 2]

        with self.subTest('ordered vocabulary'):
            similar_words_retriever = SimilarWordsRetriever(voc)

            sim_words_k = similar_words_retriever.get_similar_k(word, k=len(voc))
            sim_words_thresh = similar_words_retriever.get_similar_thresh(word, 0)

            self.assertEqual(tuple(sim_words_k['original terms']),
                             tuple(sim_words_thresh['original terms']))
            self.assertEqual(tuple(sim_words_k['score']),
                             tuple(sim_words_thresh['score']))
            self.assertEqual(tuple(sim_words_k['original terms']),
                                   tuple(voc))

            idx = [voc.index(term) for term in sim_words_k['original terms']]

            self.assertEqual(idx_sorted, idx)

        voc_scramble = voc[1:] + voc[:1]
        idx_scramble = idx_sorted[1:] + idx_sorted[:1]

        with self.subTest('scrambled vocabulary'):

            similar_words_retriever = SimilarWordsRetriever(voc_scramble)

            sim_words_k = similar_words_retriever.get_similar_k(word, k=len(voc_scramble))
            sim_words_thresh = similar_words_retriever.get_similar_thresh(word, 0)

            self.assertEqual(tuple(sim_words_k['original terms']),
                             tuple(sim_words_thresh['original terms']))
            self.assertEqual(tuple(sim_words_k['score']),
                             tuple(sim_words_thresh['score']))
            self.assertEqual(tuple(voc), tuple(sim_words_k['original terms']))

            idx = [sim_words_k['original terms'].index(term) for term in voc_scramble]

            self.assertEqual(idx_scramble, idx)

class TestProcessing(unittest.TestCase):

    def test_return(self):

        with self.subTest():
            for word in VOC_RANDOM_SYMBOLS:
                pre_word  = preprocessing_word(word)
                self.assertIsInstance(pre_word, str)
                self.assertTrue(pre_word)

        with self.subTest():
            for word in VOC_RANDOM_SENTENCE:
                pre_word  = preprocessing_word(word)
                self.assertIsInstance(pre_word, str)
                self.assertTrue(pre_word)

    def test_no_space(self):

        for word in VOC_RANDOM_SENTENCE:
            pre_word  = preprocessing_word(word)

            self.assertTrue(' ' not in pre_word)

    def test_lowercase(self):

        word = 'UPPERlowerCaSe'

        self.assertEqual(word, preprocessing_word(word, b_lower=False), 'casing should not have changed')

        self.assertEqual(word.lower(), preprocessing_word(word, b_lower=True), 'should be lowercase')

    def test_weird_space(self):

        word0 = ' a  big   space   '
        word1 = 'a_big_space'
        self.assertEqual(word1, preprocessing_word(word0), 'There should not be any double spaces')


class TestEmbedding(unittest.TestCase):
    swr = SimilarWordsRetriever([])
    def test_single_word(self):

        embedding = self.swr.get_embedding('word')
        self.assertIsNotNone(embedding, 'Embedding should return something')
        self.assertIsInstance(embedding, (list, tuple, np.ndarray), 'embedding should be vector')

    def test_multigram(self):
        embedding = self.swr.get_embedding('two words')
        self.assertIsNotNone(embedding, 'Multigrams als need to be encoded')


    def test_multiple_words(self):
        words = ['one', 'Dos Tres']
        embedding = self.swr.get_embedding(words)
        self.assertIsNotNone(embedding, 'Should be able to encode lists of words')

        self.assertEqual(len(words), len(embedding), 'Should return a list that is of same length')

    def test_normalisation(self):

        for string in VOC_ALL: #['word', 'two words', ['one', 'two', 'three']]:
            embedding = self.swr.get_embedding(string)

            norm = np.linalg.norm(embedding, axis=-1, keepdims=False)

            # check of all norm of the embedding of each word is (near) 1.
            self.assertTrue(np.all(np.isclose(norm, 1, 1e-5)), f'embedding should be normalised on unit sphere: {string}')


class TestSimilarWords(unittest.TestCase):

    def test_most_similar_return(self):
        corpus = ['one', 'two', 'three']
        word = 'four'
        sim_words = SimilarWords(corpus=corpus)

        k = 2
        retrieved_words, idx, distances = sim_words.get_similar_terms(word, k=k)

        self.assertEqual(len(retrieved_words), len(idx), 'outputs should have equal length')
        self.assertEqual(len(retrieved_words), len(distances), 'outputs should have equal length')

        for a, b in zip(distances, sorted(distances)):
            self.assertEqual(a, b, 'output should be sorted according to distances')

    def test_most_similar(self):
        """If the term is in the corpus, the most similar term should be the term itself.
        """

        corpus = ['one', 'two', 'three']

        word = corpus[-1]
        sim_words = SimilarWords(corpus=corpus)

        k = 1
        retrieved_words, _, _ = sim_words.get_similar_terms(word, k=k)

        self.assertEqual(len(retrieved_words), k, 'Only k most similar terms should be retrieved!')

        self.assertEqual(retrieved_words[0], word, 'Retrieved word should be equal if in corpus')

    def test_check_in_corpus(self):
        corpus = ['one', 'two', 'three']

        word = 'boogus'

        assert word not in corpus, 'BUG IN TEST. This test only makes sense if word not in corpus'

        sim_words = SimilarWords(corpus=corpus)
        retrieved_words, _, _ = sim_words.get_similar_terms(word, k=2)

        for retrieved_word in retrieved_words:
            self.assertIn(retrieved_word, corpus, 'retrieved word should be member of corpus')

        self.assertNotIn(word, retrieved_words, 'since word not in corpus, most similar words can not be this word')


class TestArgpartitionsort(unittest.TestCase):

    x = [5161, 235, 5831, 1831, 13813, 894]
    x = ['d', 'zxcv', 'ex', 'asdb we', ' qasdf', 'sd fw ']
    x_sorted = sorted(x)
    idx_sorted = np.argsort(x)

    def test_return(self):
        k= 3
        argpartitionsort(self.x, k)

    def test_k_too_high(self):
        """ If more similar indices are requested then there exist, it should just return as much as possible.

        Returns:
            None
        """
        k= 2*len(self.x)

        idx = argpartitionsort(self.x, k, reverse=False)
        self.assertEqual(tuple(self.idx_sorted), tuple(idx))

        idx = argpartitionsort(self.x, k, reverse=True)
        self.assertEqual(tuple(self.idx_sorted[::-1]), tuple(idx))

    def test_sorting(self):
        k= 3
        idx = argpartitionsort(self.x, k, reverse=False)

        self.assertEqual(tuple(self.idx_sorted[:k]), tuple(idx))

    def test_reverse(self):
        k = 3
        idx = argpartitionsort(self.x, k, reverse=True)

        self.assertEqual(tuple(self.idx_sorted[::-1][:k]), tuple(idx))

    def test_k(self):
        """Shouldn't crash with whatever value for k.
        We expect k to be bigger or equal than 0.

        Returns:
            None
        """
        n = len(self.x)
        for k in (1, n-2, n-1, n, n+1, 10*n):
            idx = argpartitionsort(self.x, k, reverse=False)
            self.assertGreaterEqual(k, len(idx))

            idx = argpartitionsort(self.x, k, reverse=True)
            self.assertGreaterEqual(k, len(idx))

            # self.assertEqual(tuple(self.idx_sorted[::-1][:k]), tuple(idx))

if __name__ == '__main__':
    unittest.main()
