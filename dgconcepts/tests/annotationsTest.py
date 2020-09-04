import base64
import os
from pathlib import Path

import ahocorasick as ahc
import pandas as pd
from cassis.typesystem import load_typesystem
from cassis.xmi import load_cas_from_xmi

from ..pipeline import terms
from ..pipeline import tables
from ..pipeline.cleaning import get_text_html

MEDIA_DIR = os.path.join(Path(os.path.abspath(__file__)).parents[2], 'media')
pandas_dataframe_2 = os.path.join(MEDIA_DIR, 'full_dgf_jsons_table2.csv')
encoded_xmi = open(os.path.join(Path(os.path.abspath(__file__)).parent,'test_xmi')).read()

def test_annotations():  # f is a json {"xml" : "encoded_xml"}
    with open(os.path.join(MEDIA_DIR, 'typesystem.xml'), 'rb') as x:
        typesystem = load_typesystem(x)

    cas = load_cas_from_xmi(base64.b64decode(encoded_xmi).decode('utf-8'), typesystem=typesystem)
    sofa_id = "html2textView"
    sentences = get_text_html(cas, sofa_id, tagnames=['p'])  # html or pdf get_text_pdf
    dict_v1, abvs = terms.analyzeFile(' '.join(sentences))
    terms_n_tfidf = tables.calculate_tf_idf(dict_v1, pandas_dataframe_2)
    for abv in abvs:
        terms_n_tfidf.update({abv: 1})

    terms_tf_idf = [(k, v) for k, v in terms_n_tfidf.items()]
    Token = typesystem.get_type('de.tudarmstadt.ukp.dkpro.core.api.frequency.tfidf.type.Tfidf')
    A = ahc.Automaton()
    cas_view = cas.get_view(sofa_id)
    for (key, cat) in terms_tf_idf:
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

    # sanity check of the offsets
    for item in cas_view.select("de.tudarmstadt.ukp.dkpro.core.api.frequency.tfidf.type.Tfidf"):
        assert(item.term == item.get_covered_text())