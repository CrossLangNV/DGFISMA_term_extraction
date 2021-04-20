import re
import string

from configparser import ConfigParser
from typing import Tuple, List

from cassis import Cas, TypeSystem

from .utils import lemmatize

from spacy.lang.en import English

'''
Module to perform rule based term and definition extraction from tables.
Terms and definitions are extracted from eur-lex files with table header: 'Legal references and instructions'.
'''

def rule_based_concept_extraction( NLP: English, cas: Cas, typesystem: TypeSystem, config: ConfigParser ):
    
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
                    get_terms_and_definitions( NLP, cas, typesystem, config, tag )
                    break

def get_terms_and_definitions(NLP: English, cas: Cas, typesystem: TypeSystem, config: ConfigParser, table_tag):
    
    SofaID = config[ 'Annotation' ].get( 'SOFA_ID' )
    Value_between_tagtype = config[ 'Annotation' ].get( 'VALUE_BETWEEN_TAG_TYPE' )
    
    term_cache = []  #list of all terms in the tbody tag
    for tag in cas.get_view( SofaID ).select_covered(Value_between_tagtype, table_tag):
        if tag.tagName == 'td' and "valign='top'class='table'" in tag.attributes:
            for subtag in cas.get_view( SofaID ).select_covered(Value_between_tagtype, tag):
                if subtag.tagName == 'p' and "class='ti-grseq-1'" in subtag.attributes:
                    #strip enumeration from start of a term
                    stripped_term=clean_term( subtag.get_covered_text() )
                    stripped_term=stripped_term.lower()
                    #search for offset of stripped_term in sofa:
                    offset_stripped_term=get_offset_stripped_term( stripped_term, subtag.get_covered_text() )
                    if not offset_stripped_term:
                        continue
                    term = ( stripped_term , subtag.begin+offset_stripped_term[0], subtag.begin + offset_stripped_term[1] )
                    definition = ( tag.get_covered_text().strip(), tag.begin, tag.end )
                    if not stripped_term.lower().startswith('of which'):
                        term_cache.append( stripped_term )
                    else:
                        if term_cache:
                            term_text=term_cache[-1]+ ' '+term[0]
                            term=( term_text , term[1], term[2]  )
                    annotate_terms_and_definitions( NLP, cas, typesystem, config, (term, definition) )
                    #break, because assume only one term (with class='ti-grseq-1' attribute ) in each 'td' tag
                    break

def remove_terms_and_definitions_ml( cas: Cas, typesystem: TypeSystem, config: ConfigParser, table_tag  ):
    
    defined_type = config[ 'Annotation' ].get( 'DEFINED_TYPE' )
    definition_type = config[ 'Annotation' ].get( 'DEFINITION_TYPE' )
    paragraph_type= config[ 'Annotation' ].get( 'PARAGRAPH_TYPE' )
    SofaID = config[ 'Annotation' ].get( 'SOFA_ID' )

    #remove possible concepts detected by ML algorithm in the tables
    
    for concept in cas.get_view( SofaID ).select_covered( defined_type , table_tag ):
        cas.get_view( SofaID ).remove_annotation(  concept )
        
    #remove possible definitions detected by ML algorithm in the tables
    
    for definition in cas.get_view( SofaID ).select_covered( definition_type , table_tag ):
        cas.get_view( SofaID ).remove_annotation(  definition )
        
    #remove possible paragraphs detected by ML algorithm in the tables

    for par in cas.get_view( SofaID ).select_covered( paragraph_type , table_tag ):
        cas.get_view( SofaID ).remove_annotation(  par )

        
def annotate_terms_and_definitions(NLP:English, cas: Cas, typesystem: TypeSystem, config: ConfigParser, \
                                   term_and_definition: Tuple[ Tuple[ str, int, int ], Tuple[str, int, int] ]  ):
    
    token_type = config[ 'Annotation' ].get( 'TOKEN_TYPE' )
    defined_type = config[ 'Annotation' ].get( 'DEFINED_TYPE' )
    definition_type = config[ 'Annotation' ].get( 'DEFINITION_TYPE' )
    lemma_type=config[ 'Annotation' ].get( 'LEMMA_TYPE' )
    SofaID= config[ 'Annotation' ].get( 'SOFA_ID' )
    SCORE=config['TermExtraction'].getfloat( 'TFIDF_WHITELIST' )

    Token = typesystem.get_type( token_type )
    Defined = typesystem.get_type( defined_type )
    Sentence = typesystem.get_type( definition_type )
    Lemma = typesystem.get_type( lemma_type )

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
    cas.get_view( SofaID ).add_annotation(
        Lemma(begin=term_begin, end=term_end, value=lemmatize( NLP, term ) ))
    
def clean_term( term ):
    #remove possible enumeration from the term (i.e. 1.1. something ==> something)
    
    term=term.strip()
    #remove a. from start of term
    term=re.sub( "^[\.A-Za-z\/]*?(?=\.+?\s)", "", term  )
    #remove 1.1. and 1.from start of term and 1.1*, 1.1**
    term=re.sub( "^[0-9\.\/]*?(?=\.*?[\s\\*])","",term)
    term=term.strip()
    #remove (-), (+) and (*) from start of term
    term=term.lstrip( "*. " ).strip()
    term=re.sub( "^\([-+*]\)\s*", "", term  )

    term=term.lstrip( "*. " ).strip()
    
    return term

def get_offset_stripped_term( stripped_term, original_text  ):
    #find offset of stripped term in the original text

    match=re.search(  stripped_term ,  original_text , re.IGNORECASE )

    if match:
        offset_stripped_term=(match.span()[0], match.span()[1] )
        return offset_stripped_term
    else:
        return None
