import os
import logging
import base64
import binascii
import time
from cassis.typesystem import load_typesystem
from cassis.xmi import load_cas_from_xmi
from django.http import JsonResponse
from django.conf import settings
from rest_framework.views import APIView

import spacy

from .pipeline.utils import get_sentences
from .pipeline.terms import get_terms, get_terms_defined_in_regex, remove_add_update_terms_blacklist_whitelist
from .pipeline.terms_defined import add_nsubj_dependency, add_defined_term
from .pipeline.annotations import add_terms_and_lemmas_to_cas

NLP = spacy.load('en_core_web_lg')
WHITELIST = open(os.path.join(settings.MEDIA_ROOT, 'whitelist.txt')).read().splitlines()
BLACKLIST = open(os.path.join(settings.MEDIA_ROOT, 'blacklist.txt')).read().splitlines()
MAX_LEN_NGRAM = 4
EXTRACT_SUPERGRAMS=False
TFIDF_REGEX=-2.0
TFIDF_WHITELIST=-1.0

with open(os.path.join(settings.MEDIA_ROOT, 'typesystem.xml'), 'rb') as f:
    TYPESYSTEM = load_typesystem(f)

class TermView(APIView):
        
    def post(self, request):
        
        SofaID='html2textView'
        
        start = time.time()

        f = request.data
        
        output_json={}

        try:
            decoded_cas_content = base64.b64decode(f['cas_content']).decode('utf-8')
        except binascii.Error:
            logging.info(f"could not decode the 'cas_content' field. Make sure it is in base64 encoding.")
            end = time.time()
            logging.info(end - start)
            return JsonResponse(f)
            
        cas = load_cas_from_xmi(decoded_cas_content, typesystem=TYPESYSTEM)

        try:
            cas.get_view( SofaID )
        except:
            logging.info(f"could not process the view in this CAS. Make sure it contains a {SofaID} view.")
            end = time.time()
            logging.info(end - start)
            return JsonResponse(f)

        sentences, _ = get_sentences( cas, SofaID )
        
        #get a dictionary with all detected terms and tfidf score
        terms_n_tfidf, _ = get_terms( NLP, sentences, extract_supergrams = EXTRACT_SUPERGRAMS, nMax = MAX_LEN_NGRAM )

        definitions=[definition.get_covered_text() for definition in \
        cas.get_view( SofaID ).select(  'de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence' ) ]

        #get a list of terms defined in detected definitions, via regex (i.e. in between quotes)
        terms_defined=get_terms_defined_in_regex( definitions )
        
        for term in terms_defined:
            terms_n_tfidf.update( { term: TFIDF_REGEX }  )
        
        #remove terms found in definitions between quotes from the whitelist (because these are already in dictionary terms_n_tfidf)
        whitelist=set( WHITELIST ) - set( terms_defined )
        blacklist=set( BLACKLIST )

        terms_n_tfidf = remove_add_update_terms_blacklist_whitelist( terms_n_tfidf, whitelist, blacklist, tf_idf_whitelist=TFIDF_WHITELIST )

        add_terms_and_lemmas_to_cas( NLP, cas, TYPESYSTEM, SofaID, terms_n_tfidf ) 

        add_nsubj_dependency( NLP, cas, TYPESYSTEM , SofaID , definition_type= 'de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence' )

        add_defined_term( cas, TYPESYSTEM, SofaID , definition_type= 'de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence', \
                         tf_idf_whitelist=TFIDF_WHITELIST, tf_idf_regex=TFIDF_REGEX  )

        output_json['cas_content']=base64.b64encode(  bytes( cas.to_xmi()  , 'utf-8' ) ).decode()   
        output_json[ 'content_type']=f[ 'content_type']
        
        end = time.time()
        logging.info(end - start)

        return JsonResponse(output_json)
