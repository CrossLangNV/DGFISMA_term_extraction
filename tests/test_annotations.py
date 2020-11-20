import pytest
import os
import configparser

from dgconcepts.pipeline.annotations import add_defined_term_regex, add_defined_term_custom_tf_idf_score, add_defined_term_bio_tagging

from cassis.typesystem import load_typesystem

from ._create_cas import _create_test_cas_1

MEDIA_ROOT='media'
                        
with open( os.path.join( MEDIA_ROOT, 'typesystem.xml' )  , 'rb') as f:
    TYPESYSTEM = load_typesystem(f)

config = configparser.ConfigParser()
config.read( os.path.join( MEDIA_ROOT, 'TermExtraction.config' ))

def test_add_defined_term_regex_no_fall_back():
    
    true_defined_term_found=[True,False,False,False]
    
    true_defined_terms=[(11, 20)]
    
    cas=_create_test_cas_1()
    
    for i, sentence_feature in enumerate(cas.get_view( config[ 'Annotation' ]['Sofa_ID'] ).select( config[ 'Annotation' ][ 'DEFINITION_TYPE'  ] )):
        defined_term_found=add_defined_term_regex( cas, TYPESYSTEM, config , sentence_feature, fall_back_regex=False )
        assert true_defined_term_found[i]==defined_term_found

    defined_terms=[(defined_term.begin, defined_term.end) for defined_term in cas.get_view( config[ 'Annotation' ]['Sofa_ID']  ).select( config[ 'Annotation' ][ 'DEFINED_TYPE' ] )]
    assert true_defined_terms == defined_terms
    
    
def test_add_defined_term_regex_fall_back():
    
    #if fall_back == True in add_defined_term_regex, then all detected terms via regex will be annotated with defined_type, for sentences for which no dependency on the regex is found.
    
    true_defined_term_found=[True,True,False,False]
    
    true_defined_terms=[(11, 20), (61, 65)]
    
    cas=_create_test_cas_1()
    
    for i, sentence_feature in enumerate(cas.get_view( config[ 'Annotation' ]['Sofa_ID'] ).select( config[ 'Annotation' ][ 'DEFINITION_TYPE'  ] )):
        defined_term_found=add_defined_term_regex( cas, TYPESYSTEM, config , sentence_feature, fall_back_regex=True )
        assert true_defined_term_found[i]==defined_term_found

    defined_terms=[(defined_term.begin, defined_term.end) for defined_term in cas.get_view( config[ 'Annotation' ]['Sofa_ID']  ).select( config[ 'Annotation' ][ 'DEFINED_TYPE' ] )]
    assert true_defined_terms == defined_terms
    
def test_add_defined_term_custom_tf_idf_score():
    
    #if fall_back == True in add_defined_term_custom_tf_idf_score, then all detected terms via with tf_idf_score in tf_idf_scores will be annotated with defined_type, for sentences for which no dependency on these terms is found.   
    
    tf_idf_regex=config[ 'TermExtraction' ].getfloat( 'TFIDF_REGEX' )
    tf_idf_whitelist=config[ 'TermExtraction' ].getfloat( 'TFIDF_WHITELIST' )

    true_defined_term_found=[True,True,False,False]
    
    true_defined_terms=[(11, 20), (51, 59)]
    
    cas=_create_test_cas_1()

    for i, sentence_feature in enumerate(cas.get_view( config[ 'Annotation' ]['Sofa_ID'] ).select( config[ 'Annotation' ][ 'DEFINITION_TYPE'  ] )):

        defined_term_found=add_defined_term_custom_tf_idf_score( cas, TYPESYSTEM, config , sentence_feature, \
                                                                tf_idf_scores=set([ tf_idf_regex, tf_idf_whitelist ]), fall_back=False )
        
        assert true_defined_term_found[i]==defined_term_found
        
    defined_terms=[(defined_term.begin, defined_term.end) for defined_term in cas.get_view( config[ 'Annotation' ]['Sofa_ID']  ).select( config[ 'Annotation' ][ 'DEFINED_TYPE' ] )]

    assert true_defined_terms == defined_terms
        
        
def test_add_defined_term_custom_tf_idf_score_fall_back_1():
    
    #fall back with a dependency relation in second sentence and third sentence of the the cas
    
    tf_idf_regex=config[ 'TermExtraction' ].getfloat( 'TFIDF_REGEX' )
    tf_idf_whitelist=config[ 'TermExtraction' ].getfloat( 'TFIDF_WHITELIST' )

    true_defined_term_found=[True,True,False,False]
    
    true_defined_terms=[(11, 20), (51, 59)]
    
    cas=_create_test_cas_1()

    for i, sentence_feature in enumerate(cas.get_view( config[ 'Annotation' ]['Sofa_ID'] ).select( config[ 'Annotation' ][ 'DEFINITION_TYPE'  ] )):

        defined_term_found=add_defined_term_custom_tf_idf_score( cas, TYPESYSTEM, config , sentence_feature, \
                                                                tf_idf_scores=set([ tf_idf_regex, tf_idf_whitelist ]), fall_back=True )
        
        assert true_defined_term_found[i]==defined_term_found
        
    defined_terms=[(defined_term.begin, defined_term.end) for defined_term in cas.get_view( config[ 'Annotation' ]['Sofa_ID']  ).select( config[ 'Annotation' ][ 'DEFINED_TYPE' ] )]

    assert true_defined_terms == defined_terms
    
    
def test_add_defined_term_custom_tf_idf_score_fall_back_2():
    
    #fall back with no dependency relation in second sentence and third of the cas
    
    tf_idf_regex=config[ 'TermExtraction' ].getfloat( 'TFIDF_REGEX' )
    tf_idf_whitelist=config[ 'TermExtraction' ].getfloat( 'TFIDF_WHITELIST' )

    true_defined_term_found=[True,True,False,False]
    
    true_defined_terms=[(11, 20), (43, 47), (51, 59), (61, 65)]
    
    cas=_create_test_cas_1( extra_dependencies=False )

    for i, sentence_feature in enumerate(cas.get_view( config[ 'Annotation' ]['Sofa_ID'] ).select( config[ 'Annotation' ][ 'DEFINITION_TYPE'  ] )):

        defined_term_found=add_defined_term_custom_tf_idf_score( cas, TYPESYSTEM, config , sentence_feature, \
                                                                tf_idf_scores=set([ tf_idf_regex, tf_idf_whitelist ]), fall_back=True )
        
        assert true_defined_term_found[i]==defined_term_found
        
    defined_terms=[(defined_term.begin, defined_term.end) for defined_term in cas.get_view( config[ 'Annotation' ]['Sofa_ID']  ).select( config[ 'Annotation' ][ 'DEFINED_TYPE' ] )]

    assert true_defined_terms == defined_terms
    
    
def test_add_defined_term_custom_tf_idf_score_noregex_no_whitelist():
    
    #no fall back
    
    tf_idf_regex=config[ 'TermExtraction' ].getfloat( 'TFIDF_REGEX' )
    tf_idf_whitelist=config[ 'TermExtraction' ].getfloat( 'TFIDF_WHITELIST' )

    true_defined_term_found=[True,True,True,False]
    
    true_defined_terms=[(11, 20), (51, 59), (67, 80)]
    
    cas=_create_test_cas_1()

    for i, sentence_feature in enumerate(cas.get_view( config[ 'Annotation' ]['Sofa_ID'] ).select( config[ 'Annotation' ][ 'DEFINITION_TYPE'  ] )):

        defined_term_found=add_defined_term_custom_tf_idf_score( cas, TYPESYSTEM, config , sentence_feature, \
                                                                tf_idf_scores=set(), fall_back=False )
        
        assert true_defined_term_found[i]==defined_term_found
        
    defined_terms=[(defined_term.begin, defined_term.end) for defined_term in cas.get_view( config[ 'Annotation' ]['Sofa_ID']  ).select( config[ 'Annotation' ][ 'DEFINED_TYPE' ] )]

    assert true_defined_terms == defined_terms
    
def test_add_defined_term_custom_tf_idf_score_noregex_no_whitelist_fall_back_1():
    
    #fall back with a dependency relation in second sentence and third sentence of the the cas
    
    tf_idf_regex=config[ 'TermExtraction' ].getfloat( 'TFIDF_REGEX' )
    tf_idf_whitelist=config[ 'TermExtraction' ].getfloat( 'TFIDF_WHITELIST' )

    true_defined_term_found=[True,True,True,False]
    
    true_defined_terms=[(11, 20), (51, 59), (67, 80)]
    
    cas=_create_test_cas_1(  )

    for i, sentence_feature in enumerate(cas.get_view( config[ 'Annotation' ]['Sofa_ID'] ).select( config[ 'Annotation' ][ 'DEFINITION_TYPE'  ] )):

        defined_term_found=add_defined_term_custom_tf_idf_score( cas, TYPESYSTEM, config , sentence_feature, \
                                                                tf_idf_scores=set(), fall_back=True )
        
        assert true_defined_term_found[i]==defined_term_found
        
    defined_terms=[(defined_term.begin, defined_term.end) for defined_term in cas.get_view( config[ 'Annotation' ]['Sofa_ID']  ).select( config[ 'Annotation' ][ 'DEFINED_TYPE' ] )]

    assert true_defined_terms == defined_terms
        
    
def test_add_defined_term_custom_tf_idf_score_noregex_no_whitelist_fall_back_2():
    
    #fall back with no dependency relation in second sentence and third sentence of the the cas
    
    tf_idf_regex=config[ 'TermExtraction' ].getfloat( 'TFIDF_REGEX' )
    tf_idf_whitelist=config[ 'TermExtraction' ].getfloat( 'TFIDF_WHITELIST' )

    true_defined_term_found=[True,True,True,False]
    
    true_defined_terms=[(11, 20), (43, 47), (51, 59), (61, 65), (67, 80)]
    
    cas=_create_test_cas_1( extra_dependencies=False )

    for i, sentence_feature in enumerate(cas.get_view( config[ 'Annotation' ]['Sofa_ID'] ).select( config[ 'Annotation' ][ 'DEFINITION_TYPE'  ] )):

        defined_term_found=add_defined_term_custom_tf_idf_score( cas, TYPESYSTEM, config , sentence_feature, \
                                                                tf_idf_scores=set(), fall_back=True )
        
        assert true_defined_term_found[i]==defined_term_found
        
    defined_terms=[(defined_term.begin, defined_term.end) for defined_term in cas.get_view( config[ 'Annotation' ]['Sofa_ID']  ).select( config[ 'Annotation' ][ 'DEFINED_TYPE' ] )]

    assert true_defined_terms == defined_terms
    
    
def test_add_defined_term_bio_tagging():
    
    true_defined_terms=[(16, 20), (52, 54), (67, 71)]
    true_terms=['this', 'ha', 'some']

    cas=_create_test_cas_1( )

    terms=[]
    
    for i, sentence_feature in enumerate(cas.get_view( config[ 'Annotation' ]['Sofa_ID'] ).select( config[ 'Annotation' ][ 'DEFINITION_TYPE'  ] )):

        terms+=add_defined_term_bio_tagging( cas, TYPESYSTEM, config, sentence_feature )

    assert true_terms==terms

    defined_terms=[(defined_term.begin, defined_term.end) for defined_term in cas.get_view( config[ 'Annotation' ]['Sofa_ID']  ).select( config[ 'Annotation' ][ 'DEFINED_TYPE' ] )]

    assert true_defined_terms == defined_terms
        
        
        
        
        
        


    