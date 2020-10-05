import base64
import binascii
import os
import time
from cassis.typesystem import load_typesystem
from cassis.xmi import load_cas_from_xmi
from django.conf import settings
from pipeline import tables
from pipeline import terms  # , cleaning
from pipeline.annotations import *
from pipeline.cleaning import get_text_html

start = time.time()
f = open('/Users/alina/Desktop/xiw.txt', 'r').read()
dict_v1, abvs = terms.analyzeFile(f)
#terms_n_tfidf = tables.calculate_tf_idf(dict_v1, PATH_TO_PD2)
mediaan = time.time()
print("term extraction done in " + str(mediaan - start))
# terms_n_tfidf = tables.crosscheck_unigrams(terms_n_tfidf)
print(dict_v1['ngrams'])
print(len(dict_v1['ngrams']))
end = time.time()
#print(cas.to_xmi())
#print(str(terms_n_tfidf))
print(end - start)

"""
ValueError: [E088] Text of length 6440150 exceeds maximum of 1000000. The v2.x parser and NER models require roughly 1GB of temporary memory per 100,000 characters in the input. This means long texts may cause memory allocation errors. If you're not using the parser or NER, it's probably safe to increase the `nlp.max_length` limit. The limit is in number of characters, so you can check whether your inputs are too long by checking `len(text)`.
"""