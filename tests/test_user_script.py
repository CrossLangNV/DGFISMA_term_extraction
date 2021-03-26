import json
import base64
import os
import configparser

from cassis.typesystem import load_typesystem
from cassis.xmi import load_cas_from_xmi

from user_scripts.generate_training_data_from_cas import annotations_to_tags, extract_user_annotations_from_cas
from user_scripts.utils import remove_quotations_around_terms

MEDIA_ROOT='media'
TEST_FILES='tests/files/test_files'

with open( os.path.join( MEDIA_ROOT, "typesystem_user.xml" ) , 'rb') as f:
    TYPESYSTEM = load_typesystem(f)
    
CONFIG = configparser.ConfigParser()
CONFIG.read( os.path.join( MEDIA_ROOT, 'TermExtraction.config' ) )
    
def test_extract_user_annotations_from_cas():

    with open( os.path.join( TEST_FILES ,'jsons_with_definitions/small_nested_tables_response.json' ) ) as json_file:
        response = json.load(json_file)
    
    decoded_cas_response = base64.b64decode( response[ 'cas_content' ] ).decode( 'utf-8' )

    cas = load_cas_from_xmi( decoded_cas_response , typesystem=TYPESYSTEM, trusted=True )
        
    #add user annotations
    Defined_type = TYPESYSTEM.get_type( CONFIG['Annotation_user']['DEFINED_TYPE_USER'] )
    
    cas.get_view( CONFIG[ 'Annotation' ].get( 'SOFA_ID' ) ).add_annotation( Defined_type( begin=4271 , end=4300, user="user1", datetime="Sat_Dec_5_11:01:21_2020" ) )

    cas.get_view( CONFIG[ 'Annotation' ].get( 'SOFA_ID' ) ).add_annotation( Defined_type( begin=4306 , end=4334, user="user2", datetime="Sat_Dec_5_11:01:21_2020" ) )

    cas.get_view( CONFIG[ 'Annotation' ].get( 'SOFA_ID' ) ).add_annotation( Defined_type( begin=4306 , end=4334, user="user3", datetime="Sat_Dec_5_11:01:21_2020" ) )
    
    
    Token_type=TYPESYSTEM.get_type( CONFIG['Annotation_user']['TOKEN_TYPE_USER'] )
    cas.get_view( CONFIG[ 'Annotation' ].get( 'SOFA_ID' ) ).add_annotation( Token_type( term="significant supervised entity", begin=4271 , end=4300, user="user1", datetime="Sat_Dec_5_11:01:21_2020" ) )
    
    cas.get_view( CONFIG[ 'Annotation' ].get( 'SOFA_ID' ) ).add_annotation( Token_type( term="significant supervised group", begin=4306 , end=4334, user="user1", datetime="Sat_Dec_5_11:01:21_2020" ) )

    terms, definitions, annotated_sentences = extract_user_annotations_from_cas( cas, CONFIG, users=['user1','user2'] )
    
    true_definitions=[ '‘significant supervised entity’ or ‘significant supervised group’ means an entity or group as referred to in Article 4(1)(a);']
    true_annotated_sentences=[' ★  significant supervised entity  ☆  or  ★  significant supervised group  ☆  means an entity or group as referred to in Article 4(1)(a);']
    true_terms=[ "significant supervised entity", "significant supervised group"  ]
    
    assert definitions == true_definitions
    assert annotated_sentences == true_annotated_sentences
    assert true_terms==terms
    
    
def test_annotations_to_tags():
    sentences=[
        'This is a test sentence'
    ]
    
    list_defined=[
    [ ( 0,4 ), ( 10,14 ) ],
    [ ( 0,4 ), (8,9), ( 10,14 ) ],
    [ ( 0,4 ) ]
    ]
    
    sentences_new=[
        ' ★ This ☆  is a  ★ test ☆  sentence',
        ' ★ This ☆  is  ★ a ☆   ★ test ☆  sentence',
        ' ★ This ☆  is a test sentence'
    ]
    
    for sentence, defined, sentence_new, in zip( sentences, list_defined, sentences_new ):
        assert annotations_to_tags( sentence, defined ) == sentence_new
        
        
def test_remove_quotations_around_terms():
    
    sentences=[
        " ★ `test`★ ` ",
        " ` ★ ` test`★ ` ` ",
       '‘ ★ significant supervised entity ☆ ’ or ‘ ★ significant supervised group ☆ ’ means an entity'
    ]
        
    processed_sentences=[
        '  ★ test ★   ',
        '  ★   test ★   ` ',
        ' ★  significant supervised entity  ☆  or  ★  significant supervised group  ☆  means an entity'
    ]
        
    for sentence, processed_sentence in zip( sentences, processed_sentences ):
        assert processed_sentence == remove_quotations_around_terms( sentence )