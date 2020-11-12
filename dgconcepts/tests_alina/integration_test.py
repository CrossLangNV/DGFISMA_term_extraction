'''
import base64
import os
import unittest
import spacy
from cassis.typesystem import load_typesystem
from cassis.xmi import load_cas_from_xmi
from dgconcepts.pipeline import terms
from dgconcepts.pipeline.utils import get_text
from dgconcepts.pipeline.annotations import add_terms_and_lemmas_to_cas
from dgconcepts.pipeline.metrics import calculate_tf_idf

class TestTermExtractionModules(unittest.TestCase):
    NLP = spacy.load('en_core_web_lg')
    MAX_LEN_NGRAM = 4
    encoded_xmi = open('test_xmi').read()
    def text_to_terms(self, sentences):
        """
        reproducing the term extraction pipeline
        :return: list of terms
        """
        doc_for_tf_idf = []
        all_abvs = []
        all_terms = []
        for sentence in sentences:
            doc_for_tf_idf.append(sentence)
            terms_so_far = []
            ngrams, supergrams, abvs = terms.extract_concepts(sentence, self.NLP, self.MAX_LEN_NGRAM)
            all_abvs.append(abvs)
            terms_so_far.append(ngrams)
            terms_so_far.append(supergrams)
            terms_so_far = [t for t_sublist in terms_so_far for t in t_sublist]
            for x in terms_so_far:
                all_terms.append(x)
        all_terms = list(set(all_terms))
        return doc_for_tf_idf, all_terms

    def test_annotations(self):
        with open(os.path.join('typesystem.xml'), 'rb') as x:
            typesystem = load_typesystem(x)
        cas = load_cas_from_xmi(base64.b64decode(self.encoded_xmi).decode('utf-8'), typesystem=typesystem)
        sofa_id = "html2textView"
        sentences = get_text(cas, sofa_id, tagnames=['p'])
        doc_for_tf_idf, all_terms = self.text_to_terms(sentences)  # we extract both ngrams and supergrams
        terms_n_tfidf = calculate_tf_idf(doc_for_tf_idf, self.MAX_LEN_NGRAM, list(set(all_terms)))
        self.assertIsInstance(terms_n_tfidf, dict)
        self.assertNotEqual(len(terms_n_tfidf.keys()), 0)
        cas = add_terms_and_lemmas_to_cas(self.NLP, cas, typesystem, sofa_id, [(k, v) for k, v in terms_n_tfidf.items()])
        cas_view = cas.get_view(sofa_id)
        for item in cas_view.select("de.tudarmstadt.ukp.dkpro.core.api.frequency.tfidf.type.Tfidf"):
            self.assertEqual(item.term, item.get_covered_text())
'''