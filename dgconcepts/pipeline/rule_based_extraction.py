import re
import string

from configparser import ConfigParser
from typing import Tuple, List

from cassis import Cas, TypeSystem

'''
Module to perform rule based term and definition extraction from tables.
Terms and definitions are extracted from eur-lex files with table header: 'Legal references and instructions'.
'''

def rule_based_concept_extraction( cas: Cas, typesystem: TypeSystem, config: ConfigParser ):
    
    SofaID = config[ 'Annotation' ].get( 'SOFA_ID' )
    Value_between_tagtype = config[ 'Annotation' ].get( 'VALUE_BETWEEN_TAG_TYPE' )

    for tag in cas.get_view(SofaID).select( Value_between_tagtype ):
        if tag.tagName == 'tbody':
            for tag_body in cas.get_view(SofaID).select_covered( Value_between_tagtype, tag ):
                if tag_body.tagName == 'p' and tag_body.attributes =="class='tbl-hdr'" \
                and tag_body.get_covered_text().strip() == 'Legal references and instructions':  
                    #if one of the p's inside the tbody tag contains "Legal references and instructions", then the tbody tag is one of the tables
                    #to process via rule based system                    
                    remove_terms_and_definitions_ml( cas, typesystem, config, tag  )
                    get_terms_and_definitions( cas, typesystem, config, tag )
                    break

def get_terms_and_definitions(cas: Cas, typesystem: TypeSystem, config: ConfigParser, table_tag):
    
    SofaID = config[ 'Annotation' ].get( 'SOFA_ID' )
    Value_between_tagtype = config[ 'Annotation' ].get( 'VALUE_BETWEEN_TAG_TYPE' )
    
    term_cache = []  #list of all terms in the tbody tag
    for tag in cas.get_view( SofaID ).select_covered(Value_between_tagtype, table_tag):
        if tag.tagName == 'td' and "valign='top'class='table'" in tag.attributes:
            for subtag in cas.get_view( SofaID ).select_covered(Value_between_tagtype, tag):
                if subtag.tagName == 'p' and "class='ti-grseq-1'" in subtag.attributes:
                    #strip enumeration from start of a term
                    stripped_term=clean_term( subtag.get_covered_text() )
                    term = ( stripped_term , subtag.begin, subtag.end )
                    definition = ( tag.get_covered_text().strip(), tag.begin, tag.end )
                    if not stripped_term.lower().startswith('of which'):
                        term_cache.append( stripped_term )
                    else:
                        if term_cache:
                            term_text=term_cache[-1]+ ' '+term[0]
                            term=( term_text , term[1], term[2]  )
                    annotate_terms_and_definitions( cas, typesystem, config, (term, definition) )
                    #break, because assume only one term (with class='ti-grseq-1' attribute ) in each 'td' tag
                    break

def remove_terms_and_definitions_ml( cas: Cas, typesystem: TypeSystem, config: ConfigParser, table_tag  ):
    
    defined_type = config[ 'Annotation' ].get( 'DEFINED_TYPE' )
    definition_type = config[ 'Annotation' ].get( 'DEFINITION_TYPE' )
    SofaID = config[ 'Annotation' ].get( 'SOFA_ID' )

    #remove possible concepts detected by ML algorithm in the tables
    
    for concept in cas.get_view( SofaID ).select_covered( defined_type , table_tag ):
        cas.get_view( SofaID ).remove_annotation(  concept )
        
    #remove possible definitions detected by ML algorithm in the tables
    
    for definition in cas.get_view( SofaID ).select_covered( definition_type , table_tag ):
        cas.get_view( SofaID ).remove_annotation(  definition )

        
def annotate_terms_and_definitions(cas: Cas, typesystem: TypeSystem, config: ConfigParser, \
                                   term_and_definition: Tuple[ Tuple[ str, int, int ], Tuple[str, int, int] ]  ):
    
    token_type = config[ 'Annotation' ].get( 'TOKEN_TYPE' )
    defined_type = config[ 'Annotation' ].get( 'DEFINED_TYPE' )
    definition_type = config[ 'Annotation' ].get( 'DEFINITION_TYPE' )
    SofaID= config[ 'Annotation' ].get( 'SOFA_ID' )
    SCORE=config['TermExtraction'].getfloat( 'TFIDF_WHITELIST' )

    Token = typesystem.get_type( token_type )
    Defined = typesystem.get_type( defined_type )
    Sentence = typesystem.get_type( definition_type )
    
    term = term_and_definition[0][0]
    term_begin = term_and_definition[0][1]
    term_end = term_and_definition[0][2]
    definition_begin = term_and_definition[1][1]
    definition_end = term_and_definition[1][2]
    cas.get_view( SofaID ).add_annotation(
        Token(begin=term_begin, end=term_end, tfidfValue=SCORE, term=term.strip().lower()  ))
    cas.get_view( SofaID ).add_annotation(
        Defined(begin=term_begin, end=term_end ))
    cas.get_view( SofaID ).add_annotation(
        Sentence(begin=definition_begin, end=definition_end, id='definition' ))
    
def clean_term( term ):
    #remove possible enumeration from the term (i.e. 1.1. something ==> something)
    
    term=term.strip()
    #remove a. from start of term
    term=re.sub( "^[\.A-Za-z\/]*?(?=\.+?\s)", "", term  )
    #remove 1.1. and 1.from start of term and 1.1*, 1.1**
    term=re.sub( "^[0-9\.\/]*?(?=\.*?[\s\\*])","",term)
    term=term.strip()
    #remove (-), (+) and (*) from start of term
    term=re.sub( "^\([-+*]\)\s*", "", term  )

    term=term.lstrip( "*. " ).strip()
    
    return term
