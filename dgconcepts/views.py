import base64
import os
import time
from cassis.typesystem import load_typesystem
from cassis.xmi import load_cas_from_xmi
from django.conf import settings
from django.http import HttpResponse
from rest_framework.views import APIView
from .pipeline import tables
from .pipeline import terms  # , cleaning
from .pipeline.annotations import add_terms_to_cas
from .pipeline.cleaning import get_text_html


class TermView(APIView):

    def post(self, request):
        with open(os.path.join(settings.MEDIA_ROOT, 'typesystem.xml'), 'rb') as f:
            typesystem = load_typesystem(f)

        start = time.time()
        f = request.data  # the input is a json with 'content' and 'content_type'
        cas = load_cas_from_xmi(base64.b64decode(f['cas_content']).decode('utf-8'), typesystem=typesystem)
        sofa_id = "html2textView"
        sentences = get_text_html(cas, sofa_id, tagnames=['p'])
        dict_v1, abbvs = terms.analyzeFile(' '.join(sentences))
        dict_v2 = tables.check_abbvs(dict_v1, abbvs)
        terms_n_tfidf = tables.calculate_tf_idf(dict_v2)
        cas = add_terms_to_cas(cas, typesystem, sofa_id, [(k, v) for k, v in terms_n_tfidf.items()])
        cas_string = base64.b64encode(bytes(cas.to_xmi(), 'utf-8'))
        end = time.time()
        f['cas_content'] = cas_string
        print(end - start)
        return HttpResponse(str(f))