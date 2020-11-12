from dgconcepts.pipeline.metrics import calculate_tf_idf
from dgconcepts.pipeline.terms import parse_doc, get_ngrams_supergrams, validate_term
import unittest
import _pickle as cPickle
import spacy

class TestTermExtractionModules(unittest.TestCase):
    nlp = spacy.load('en_core_web_lg')

    '''
    Code for dumping the SpaCy Doc object:
    ---

    import spacy
    nlp = spacy.load('en_core_web_lg')
    t = 'whereas , given the trend towards intensification of beef and veal production , premiums for stockfarming should be limited with regard to the forage capacity of each holding in relation to the numbers and species of animals held ; whereas , to avoid excessively intensive types of production , the grant of such premiums should be subject to compliance with a maximum stocking density on the holding ; whereas , however , the situation of small producers should be taken into consideration ;'
    doc = nlp(t)
    with open(r"spacy_doc.pickle", "wb") as d:
        cPickle.dump(doc, d)

    '''

    def test_tf_idf(self):
        corpus = ['the grant of such premiums should be subject to compliance with a maximum stocking density on the holding',' however the situation of small producers should be taken into consideration']
        max_len_ngram = 2
        term = 'stocking density'
        tf_idf_dic = calculate_tf_idf(corpus, max_len_ngram, ['stocking density'])
        self.assertIn(term, tf_idf_dic.keys())
        self.assertIsInstance(tf_idf_dic, dict)
        self.assertIsInstance(tf_idf_dic[term], float)
        self.assertNotEqual(len(tf_idf_dic.keys()), 0)
        self.assertTrue(1 <= tf_idf_dic[term]  <= 2)

    def test_get_ngrams_supergrams_for_unigrams(self):

        s = 'the numbers and species of animals'

        doc = self.nlp(s)
        tree = parse_doc(doc) # in this context it is safe to assume the model works properly
        nMax = 1
        expected_ngrams = ['numbers', 'species','animals']
        ngrams, supergrams = get_ngrams_supergrams(tree, nMax)
        self.assertIsInstance(ngrams, list)
        self.assertIsInstance(supergrams, list)
        self.assertNotEqual(len(ngrams), 0)
        self.assertEqual(len(supergrams), 0)
        for term in ngrams:
            self.assertIn(term, expected_ngrams)

    def test_get_ngrams_supergrams_for_bigrams(self):

        s = 'as regards the suckler cow premium, it is appropriate to maintain individual ceilings for producers'
        nlp = spacy.load('en_core_web_lg')
        doc = self.nlp(s)
        tree = parse_doc(doc) # in this context it is safe to assume the model works properly
        nMax = 2
        expected_ngrams = ['suckler cow', 'cow premium','individual ceilings', 'producers', 'ceilings', 'cow', 'premium']
        ngrams, supergrams = get_ngrams_supergrams(tree, nMax)
        self.assertIsInstance(ngrams, list)
        self.assertIsInstance(supergrams, list)
        self.assertNotEqual(len(ngrams), 0)
        self.assertNotEqual(len(supergrams), 0)
        for term in ngrams:
            self.assertIn(term, expected_ngrams)

    def test_get_ngrams_supergrams_for_termfree_text(self):

        s = 'therefore obviously testing this'
        doc = self.nlp(s)
        tree = parse_doc(doc)  # in this context it is safe to assume the model works properly
        nMax = 4
        ngrams, supergrams = get_ngrams_supergrams(tree, nMax)
        self.assertIsInstance(ngrams, list)
        self.assertIsInstance(supergrams, list)
        self.assertEqual(ngrams, supergrams)
        self.assertEqual(len(ngrams), 0)
        self.assertEqual(len(supergrams), 0)

    def test_parse_doc_into_tree(self):
        expected_tree = {'the situation of small producers', 'compliance', 'intensification of beef and veal production', 'the forage capacity of each holding in relation to the numbers and species of animals held', 'animals', 'intensification', 'premiums', 'consideration', 'relation', 'excessively intensive types', 'production', 'regard to the forage capacity of each holding in relation to the numbers and species of animals held', 'species', 'a maximum stocking density on the holding', 'excessively intensive types of production', 'the numbers', 'such premiums', 'beef and veal production', 'the holding', 'the situation', 'the numbers and species of animals', 'regard', 'the forage capacity', 'the grant', 'relation to the numbers and species of animals', 'premiums for stockfarming', 'stockfarming', 'small producers', 'the trend towards intensification of beef and veal production', 'the grant of such premiums', 'the trend', 'compliance with a maximum stocking density on the holding', 'a maximum stocking density'}
        with open(r"spacy_doc.pickle", "rb") as d:
            doc = cPickle.load(d)
        tree = parse_doc(doc)
        for leaf in tree:
            self.assertIsInstance(leaf, spacy.tokens.span.Span)
        output_tree = set()
        for leaf in tree:
            output_tree.add(leaf.text)
        self.assertEqual(expected_tree, output_tree)

    def test_validate_term(self):
        with open(r"spacy_doc.pickle", "rb") as d:
            doc = cPickle.load(d)
        expected_terms = {'intensification', 'compliance', 'premiums', 'small producers', 'regard', 'consideration', 'relation', 'animals', 'stockfarming', 'species', 'production'}
        output_terms = set()
        for np in doc.noun_chunks:
            if validate_term(np):
                output_terms.add(np.text)
        self.assertEqual(expected_terms, output_terms)