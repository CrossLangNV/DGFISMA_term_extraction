import base64
import binascii
import os
import time
from cassis.typesystem import load_typesystem
from cassis.xmi import load_cas_from_xmi
from django.conf import settings
from django.http import HttpResponse
from rest_framework.views import APIView
from .pipeline import tables
from .pipeline import terms  # , cleaning
from .pipeline.annotations import *
from .pipeline.cleaning import get_text_html

PATH_TO_PD2 = os.path.join(settings.MEDIA_ROOT, 'full_dgf_jsons_table2.csv')
PATH_TO_VOC = os.path.join(settings.MEDIA_ROOT, 'fisma-voc-filtered.csv')

class TermView(APIView):

    def post(self, request):
        with open(os.path.join(settings.MEDIA_ROOT, 'typesystem.xml'), 'rb') as f:
            typesystem = load_typesystem(f)

        start = time.time()
        f = request.data  # the input is a json with 'content' and 'content_type'
        try:
            decoded_cas_content = base64.b64decode(f['cas_content']).decode('utf-8')
        except binascii.Error:
            print(f"could not decode the 'cas_content' field. Make sure it is in base64 encoding.")

        cas = load_cas_from_xmi(decoded_cas_content, typesystem=typesystem)  # check the format
        sofa_id = "html2textView"
        sentences = get_text_html(cas, sofa_id, tagnames=['p'])  # html or pdf get_text_pdf
        dict_v1, abvs = terms.analyzeFile(' '.join(sentences))
        terms_n_tfidf = tables.calculate_tf_idf(dict_v1, PATH_TO_PD2)
        for abv in abvs:
            terms_n_tfidf.update({abv : 1})
        # terms_n_tfidf = tables.crosscheck_unigrams(terms_n_tfidf)
        cas = add_terms_to_cas(cas, typesystem, sofa_id, [(k, v) for k, v in terms_n_tfidf.items()])
        # cas = annotate_voc(cas, typesystem, sofa_id)  # cross check with terms and annotations
        cas_string = base64.b64encode( bytes( cas.to_xmi() , 'utf-8' ) ).decode()
        if 'update_voc' in f.keys():
            tables.update_voc(dict_v1, PATH_TO_VOC)
        if 'update_pd2' in f.keys():
            tables.update_pd2(dict_v1, PATH_TO_PD2)
        end = time.time()
        f['cas_content'] = cas_string
        print(end - start)
        return HttpResponse(f)