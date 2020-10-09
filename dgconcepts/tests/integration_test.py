import base64
import os
import ahocorasick as ahc
from cassis.typesystem import load_typesystem
from cassis.xmi import load_cas_from_xmi
from django.conf import settings
from ..pipeline import terms
from ..pipeline.metrics import *
from ..pipeline.cleaning import get_text_html
from ..pipeline.annotations import add_terms_and_lemmas_to_cas

import unittest
import spacy

class TestTermExtractionModules(unittest.TestCase):
    NLP = spacy.load('en_core_web_lg')
    MAX_LEN_NGRAM = 4
    encoded_xmi = open('test_xmi').read()  # amend the path

    def test_annotations(self):  # f is a json {"xml" : "encoded_xml"}
        with open(os.path.join(settings.MEDIA_ROOT, 'typesystem.xml'), 'rb') as x:
            typesystem = load_typesystem(x)

        cas = load_cas_from_xmi(base64.b64decode(self.encoded_xmi).decode('utf-8'), typesystem=typesystem)
        sofa_id = "html2textView"
        sentences = get_text_html(cas, sofa_id, tagnames=['p'])  # html or pdf get_text_pdf
        terms_n_tfidf = terms.extract_concepts(sentences, self.NLP, self.MAX_LEN_NGRAM)
        self.assertIsInstance(terms_n_tfidf, dict)
        self.assertNotEquals(len(terms_n_tfidf.keys()), 0)

        cas = add_terms_and_lemmas_to_cas(self.NLP, cas, typesystem, sofa_id, [(k, v) for k, v in terms_n_tfidf.items()])

        Token = typesystem.get_type('de.tudarmstadt.ukp.dkpro.core.api.frequency.tfidf.type.Tfidf')
        A = ahc.Automaton()
        cas_view = cas.get_view(sofa_id)
        for (key, cat) in terms_n_tfidf:
            A.add_word(key.strip(), (cat, key.strip()))
        A.make_automaton()

        for tag in cas_view.select("com.crosslang.uimahtmltotext.uima.type.ValueBetweenTagType"):
            if tag.tagName in set('p'):
                text = tag.get_covered_text()
                try:
                    for end_index, (tfidf, term) in A.iter(text):
                        start_index = end_index - len(term) + 1
                        cas_view.add_annotation(
                            Token(begin=tag.begin + start_index, end=tag.begin + end_index + 1, tfidfValue=tfidf,
                                  term=term))
                except:
                    continue

        for item in cas_view.select("de.tudarmstadt.ukp.dkpro.core.api.frequency.tfidf.type.Tfidf"):
            self.assertEquals(item.term, item.get_covered_text())