import pytest
import configparser
import spacy
import os

from dgconcepts.pipeline.terms import TermExtractor

MEDIA_ROOT='tests/test_files'

config = configparser.ConfigParser()
config.read( os.path.join( MEDIA_ROOT, 'TermExtraction.config' ))

NLP = spacy.load( config['TermExtraction'].get( 'SPACY_MODEL' ) )

def test_get_terms():
    
    sentences=['This is the first sentence', 'This is the second sentence', "European Union (EU)"  ]
    
    true_terms_n_tfidf={'european union': 1.6931,
  'first sentence': 1.6931,
  'eu': 1.6931,
  'second sentence': 1.6931}
    true_abvs=[('EU', 'European Union')]
    
    termextractor=TermExtractor( NLP, extract_supergrams = config[ 'TermExtraction' ].getboolean( 'EXTRACT_SUPERGRAMS' ), \
                               nMax = config[ 'TermExtraction' ].getint( 'MAX_LEN_NGRAM' ))
    terms_n_tfidf, abvs = termextractor.get_terms( sentences  )
    
    assert terms_n_tfidf == true_terms_n_tfidf
    assert abvs == true_abvs