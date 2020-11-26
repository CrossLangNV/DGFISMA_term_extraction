import os
import configparser
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

from .pipeline.inference import concept_extraction
from .pipeline.terms_defined.terms_defined_bio_tagging import TrainedBertBIOTagger

CONFIG = configparser.ConfigParser()
CONFIG.read( os.path.join( settings.MEDIA_ROOT, "TermExtraction.config"  ))

NLP = spacy.load( CONFIG['TermExtraction'].get( 'SPACY_MODEL' ) )

WHITELIST = open( os.path.join( settings.MEDIA_ROOT, "whitelist.txt" )).read().rstrip( "\n" ).split( "\n" )
BLACKLIST = open( os.path.join( settings.MEDIA_ROOT, "blacklist.txt" )).read().rstrip( "\n" ).split( "\n" )

SOFA_ID=CONFIG[ 'Annotation' ].get( 'SOFA_ID' )

#load the trained bio tagger
if CONFIG['DefinedTerm'].getboolean( 'BERT_BIO_TAGGING' ):
    TRAINED_BERT_BIO_TAGGER=TrainedBertBIOTagger( os.path.join( settings.MODEL_ROOT, CONFIG[ 'BertBIOTagger' ][ 'PATH_MODEL_DIR' ] ) ) 
    TRAINED_BERT_BIO_TAGGER.load_model( )
else:
    TRAINED_BERT_BIO_TAGGER=None
            
with open(os.path.join(settings.MEDIA_ROOT, 'typesystem.xml'), 'rb') as f:
    TYPESYSTEM = load_typesystem(f)
    
class TermView(APIView):
        
    def post(self, request):
                
        print( "TermView post received." )
        start = time.time()

        f = request.data
        
        output_json={}
        
        try:
            decoded_cas_content = base64.b64decode(f['cas_content']).decode('utf-8')
        except binascii.Error:
            print(f"could not decode the 'cas_content' field. Make sure it is in base64 encoding.")
            return JsonResponse(f)
            
        cas = load_cas_from_xmi(decoded_cas_content, typesystem=TYPESYSTEM)

        try:
            cas.get_view( SOFA_ID )
        except:
            print(f"could not process the view in this CAS. Make sure it contains a { SOFA_ID } view.")
            return JsonResponse(f)

        concept_extraction( NLP, TRAINED_BERT_BIO_TAGGER, cas, TYPESYSTEM, CONFIG, ( WHITELIST, BLACKLIST ) ) 
            
        output_json['cas_content']=base64.b64encode(  bytes( cas.to_xmi()  , 'utf-8' ) ).decode()   
        output_json['content_type']=f[ 'content_type']
        
        end = time.time()
        print( f"Concept extraction took {end-start} seconds." )

        return JsonResponse(output_json)
