import base64
import binascii
import time
from cassis.typesystem import load_typesystem
from cassis.xmi import load_cas_from_xmi
from django.http import JsonResponse
from django.conf import settings
from rest_framework.views import APIView
import spacy
from .pipeline import tables
from .pipeline import terms
from .pipeline.annotations import *
from .pipeline.cleaning import get_text_html
import os
from .pipeline.utils import *

NLP = spacy.load('en_core_web_lg')
WHITELIST = open(os.path.join(settings.MEDIA_ROOT, 'whitelist.csv')).read().splitlines()
BLACKLIST = open(os.path.join(settings.MEDIA_ROOT, 'blacklist.csv')).read().splitlines()
MAX_LEN_NGRAM = 4

class TermView(APIView):
    def launchTermExtraction(self, sentences, f):
        start = time.time()
        all_terms= []
        all_abvs = []
        doc_for_tf_idf = []
        for sentence in sentences:
            doc_for_tf_idf.append(sentence)
            terms_so_far = []
            ngrams, supergrams, abvs = terms.extractConcepts(sentence, NLP, MAX_LEN_NGRAM)
            all_abvs.append(abvs)
            terms_so_far.append(ngrams)
            if 'extract_supergrams' in f.keys():
                terms_so_far.append(supergrams)
            terms_so_far = [t for t_sublist in terms_so_far for t in t_sublist]
            for x in terms_so_far:
                all_terms.append(x)

        terms_n_tfidf = tables.recalculate_tf_idf(doc_for_tf_idf, list(set(all_terms)),MAX_LEN_NGRAM)
        all_abvs = [abv for abvs_sublist in all_abvs for abv in abvs_sublist]
        # all_terms = crosscheck_white_black_lists(all_terms)
        termTime = time.time()
        print('Terms extracted in: ')
        print(termTime - start)

        for abv in all_abvs:
            terms_n_tfidf.update({abv : 1.0})

        return terms_n_tfidf

    def post(self, request):
        start = time.time()

        with open(os.path.join(settings.MEDIA_ROOT, 'typesystem.xml'), 'rb') as f:
            typesystem = load_typesystem(f)
        f = request.data
        try:
            decoded_cas_content = base64.b64decode(f['cas_content']).decode('utf-8')
        except binascii.Error:
            print(f"could not decode the 'cas_content' field. Make sure it is in base64 encoding.")

        cas = load_cas_from_xmi(decoded_cas_content, typesystem=typesystem)  # check the format
        sofa_id = "html2textView"
        sentences = get_text_html(cas, sofa_id, tagnames=['p'])  # html or pdf get_text_pdf
        terms_n_tfidf = self.launchTermExtraction(sentences, f)
        cas = add_terms_and_lemmas_to_cas(NLP, cas, typesystem, sofa_id, [(k, v) for k, v in terms_n_tfidf.items()])
        cas_string = base64.b64encode( bytes( cas.to_xmi() , 'utf-8' ) ).decode()
        end = time.time()
        f['cas_content'] = cas_string
        print(end - start)
        return JsonResponse(f)