import ahocorasick as ahc
from cassis import Cas, TypeSystem
from typing import List, Tuple, Set, Generator
import re
import string

from configparser import ConfigParser

from .terms_defined_regex import process_definitions_regex
from .utils import deepest_child, is_token, select_covered, select_covering

def add_terms_and_lemmas_to_cas( NLP, cas: Cas, typesystem: TypeSystem, config: ConfigParser ,
                                terms_tf_idf: dict ) -> Cas:
    '''
    Given a cas and its typesystem, this function adds terms and associated tf_idf score (terms_tf_idf) to a given view (SofaID) as type.Tfidf. Annotations will only be added to ValueBetweenTagType elements with tag.tagName in the set tagnames. 
    NLP is the SpaCy model for the extraction of lemmas per token
    '''
    
    SofaID=config[ 'Annotation' ].get( 'SOFA_ID' )
    tagnames=set(config[ 'Annotation' ].get( 'TAG_NAMES' ))
    value_between_tagtype=config[ 'Annotation' ].get( 'VALUE_BETWEEN_TAG_TYPE' )
    token_type=config[ 'Annotation' ].get( 'TOKEN_TYPE' )
    lemma_type=config[ 'Annotation' ].get( 'LEMMA_TYPE' )
    
    Token = typesystem.get_type( token_type )
    Lemma = typesystem.get_type( lemma_type )
       
    A = ahc.Automaton()
    cas_view = cas.get_view(SofaID)
    for term in terms_tf_idf.keys():
        if not term.strip():
            continue
        A.add_word(term, ( terms_tf_idf[term], term ) )
    A.make_automaton()

    for tag in cas_view.select( value_between_tagtype ):
        if tag.tagName in set(tagnames):
            
            if not deepest_child(  cas, SofaID, tag, tagnames=tagnames, value_between_tagtype=value_between_tagtype ):
                continue
            
            text = tag.get_covered_text().lower()
            for end_index, (tfidf, term) in A.iter(text):
                if not term:
                    continue
                start_index = end_index - (len(term) - 1)
                #print( start_index, end_index, term, text  )
                if is_token(start_index, end_index, text):
                    lemmas = []
                    for word in NLP(term):
                        lemmas.append(word.lemma_)
                    term_lemmas = ' '.join(lemmas)
                    cas_view.add_annotation(
                        Token(begin=tag.begin + start_index, end=tag.begin + end_index + 1, tfidfValue=tfidf,
                              term=term))
                    cas_view.add_annotation(
                        Lemma(begin=tag.begin + start_index, end=tag.begin + end_index + 1, value=term_lemmas))


def add_dependency_annotation( cas:Cas, typesystem: TypeSystem, config: ConfigParser, defined_terms:Generator[ List[ Tuple[ str, int, int ] ], None, None ]):

    SofaID=config[ 'Annotation' ].get( 'SOFA_ID' )
    definition_type=config[ 'Annotation' ].get( 'DEFINITION_TYPE' )
    dependency_type=config[ 'Annotation' ].get( 'DEPENDENCY_TYPE' )
    
    definitions=list( cas.get_view( SofaID ).select( definition_type ))
    
    defined_terms=list( defined_terms )
    
    #Sanity check: for each definition, there should be a list of terms (or empty list), with the offsets, that are defined by the definition.
    #This list of terms should either be detected by bert bio tagger, or dependency parser.
    assert len( definitions ) == len( defined_terms )
    
    dpdc = typesystem.get_type( dependency_type )
    
    for defined_terms_sentence, definition in zip( defined_terms, definitions ):
        for defined_term in defined_terms_sentence:
            cas.get_view( SofaID ).add_annotation( dpdc( begin=definition.begin+defined_term[1] , end=definition.begin+defined_term[2], \
                                                                   DependencyType='nsubj' ))

            
def add_defined_term_annotation( cas:Cas, typesystem: TypeSystem, config: ConfigParser ):
    
    #sentence_feature=list( cas.get_view( config[ 'Annotation' ]['Sofa_ID'] ).select( config[ 'Annotation' ][ 'DEFINITION_TYPE'  ] ))[0]
    #for bert:
    #fall_back_regex=False
    #fall_back=False  #fall_back to tf idf
    #prioritize_regex=True
    #prioritize_whitelist=False
    #prioritize_tf_idf=False
    
    terms_bio_tagging=[]
    
    for sentence_feature in cas.get_view( config[ 'Annotation' ].get( 'SOFA_ID' ) ).select( config[ 'Annotation' ].get( 'DEFINITION_TYPE' ) ):
    
        defined_term_found=False

        if config[ 'DefinedTerm' ].getboolean( 'USE_REGEX' ):

            found=add_defined_term_regex( cas, typesystem , config , sentence_feature, \
                                         fall_back_regex=config[ 'DefinedTerm' ].getboolean( 'FALLBACK_TO_REGEX' ) )
            defined_term_found=found
            
        if config[ 'DefinedTerm' ].getboolean( 'USE_WHITELIST' ) and not defined_term_found:

            found=add_defined_term_custom_tf_idf_score( cas, typesystem, config, sentence_feature, \
                                                       tf_idf_scores=set([ config[ 'TermExtraction' ].getfloat( 'TFIDF_WHITELIST' ),\
                                                                          config[ 'TermExtraction' ].getfloat( 'TFIDF_REGEX' ) ]), \
                                                       fall_back=config[ 'DefinedTerm' ].getboolean( 'FALLBACK_TO_WHITELIST' ) )
            defined_term_found=found

        if config[ 'DefinedTerm' ].getboolean( 'USE_TF_IDF' ) and not defined_term_found:

            found=add_defined_term_custom_tf_idf_score( cas, typesystem, config, sentence_feature, \
                                                       tf_idf_scores=set(), \
                                                       fall_back=config[ 'DefinedTerm' ].getboolean( 'FALLBACK_TO_TF_IDF' ) )
            defined_term_found=found


        if config[ 'DefinedTerm' ].getboolean( 'USE_BERT' ) and config[ 'DefinedTerm' ].getboolean( 'BERT_BIO_TAGGING' ) and not defined_term_found:

            terms_bio_tagging+=add_defined_term_bio_tagging( cas, typesystem, config, sentence_feature )
        
    return terms_bio_tagging            


#helper functions:
            
def add_defined_term_regex( cas, typesystem , config , sentence_feature, fall_back_regex=True ):
    
    Defined_type=typesystem.get_type( config[ 'Annotation' ].get( 'DEFINED_TYPE' ) )
    SofaID= config[ 'Annotation' ].get( 'SOFA_ID' )

    sentence=sentence_feature.get_covered_text()
    
    if sentence:
    
        defined_terms_regex=list(process_definitions_regex(  [  sentence ] ))[0]

    else:
        
        return False
        
    defined_terms_regex=[( term[0] , term[1]+sentence_feature.begin, term[2]+sentence_feature.begin ) for term in defined_terms_regex]
    defined_terms_regex=set( defined_terms_regex )
    
    if not defined_terms_regex:
        
        return False
    
    rejected_terms=[]
    defined_term_found=False
    for term in select_covered( cas, SofaID, config[ 'Annotation' ].get( 'TOKEN_TYPE' ), sentence_feature  ):
        
        if (term.term, term.begin, term.end) in defined_terms_regex: #this means it is one of the detected regexes
            
            #only annotate the ones with overlapping dependency type
            if has_dependency_annotation(cas, config, term, sentence_feature ):
                cas.get_view( SofaID ).add_annotation( Defined_type( begin=term.begin , end=term.end ) )
                defined_term_found=True
            elif fall_back_regex:
                rejected_terms.append(term)
    
    #only fall-back to annotation of all terms found via regex if all terms found via regex where rejected
    if rejected_terms and fall_back_regex and not defined_term_found:
        for rejected_term in rejected_terms:
            cas.get_view( SofaID ).add_annotation( Defined_type( begin=rejected_term.begin, end=rejected_term.end )  )
            defined_term_found=True
    
    #you want to keep track of the state
    return defined_term_found

            
def add_defined_term_custom_tf_idf_score( cas, typesystem, config, sentence_feature, tf_idf_scores=set([-1.0,-2.0]), fall_back=True ):
    
    Defined_type = typesystem.get_type( config[ 'Annotation' ].get( 'DEFINED_TYPE' ) )
    SofaID = config[ 'Annotation' ].get( 'SOFA_ID' )
    
    rejected_terms=[]
    defined_term_found=False
    
    for term in select_covered( cas, SofaID, config[ 'Annotation' ].get( 'TOKEN_TYPE' ), sentence_feature  ):
        
        if tf_idf_scores and term.tfidfValue not in tf_idf_scores: #only interested in terms with specific scores ( whitelisted and regex detected terms )
            continue
        
        #check if it is the 'longest' whitelisted term at that location...
        if is_longest_term( cas, config, term, tf_idf_scores ):
            if has_dependency_annotation( cas, config, term, sentence_feature ):
                cas.get_view( SofaID ).add_annotation( Defined_type( begin=term.begin, end=term.end )  )
                defined_term_found=True
            elif fall_back:
                rejected_terms.append( term )
                
    if rejected_terms and fall_back and not defined_term_found:
        for rejected_term in rejected_terms:
            if is_longest_term( cas, config, term, tf_idf_scores ):
                cas.get_view( SofaID ).add_annotation( Defined_type( begin=rejected_term.begin, end=rejected_term.end )  )
                defined_term_found=True
    
    return defined_term_found


def add_defined_term_bio_tagging( cas, typesystem, config, sentence_feature ):
    
    
    Defined_type = typesystem.get_type( config[ 'Annotation' ].get( 'DEFINED_TYPE' ) )
    SofaID = config[ 'Annotation' ].get( 'SOFA_ID' )
    
    detected_terms=[]
    
    for dependency in select_covered( cas, SofaID, config[ 'Annotation' ].get( 'DEPENDENCY_TYPE' ), sentence_feature  ):
        
        if not defined_annotation_exists( cas, config, sentence_feature, dependency.begin, dependency.end ):
            
            detected_terms.append(dependency.get_covered_text().lower())
            
            cas.get_view( SofaID ).add_annotation( Defined_type( begin=dependency.begin, end=dependency.end )  )
            
    return detected_terms #need to annotate these terms in the cas as tfidf
        
        
#helper functions
def has_dependency_annotation( cas, config, term_feature, sentence_feature ):
    
    set1=set(list( range( term_feature.begin, term_feature.end ) ))

    for dependency_feature in cas.get_view( config[ 'Annotation' ][ 'Sofa_ID' ] ).select( config[ 'Annotation' ][ 'DEPENDENCY_TYPE' ]  ):
        set2=set( list( range( dependency_feature.begin, dependency_feature.end ) ) )
        #check if term_feature has some overlapping with a dependency feature ==> if it does, considered having a dependency annotation
        if set1.intersection( set2 ):
            return True
    
    return False

def defined_annotation_exists( cas:Cas, config:ConfigParser, sentence_feature, begin:int, end:int ):
    
    SofaID = config[ 'Annotation' ].get( 'SOFA_ID' )
    
    for defined_type in select_covered( cas, SofaID, config[ 'Annotation' ].get( 'DEFINED_TYPE' ), sentence_feature  ):
        if defined_type.begin == begin and defined_type.end==end:
            return True
        
    return False
        
def is_longest_term(cas:Cas, config:ConfigParser , term , tf_idf_scores: set() ) -> bool:

    '''
    Function checks if a tfidf annotation (term) is not part of a longer tfidf annotation. 
    Only covering tfidf annotations with specific score (tf_idf_scores) are considered (if tf_idf_score_flag is set to True).
    When tf_idf_score_flag is set to False, all covering tfidf annotations will be considered to determine if it is the longest term.
    
    :param cas: Cas.
    :param config: ConfigParser. Configuration file.
    :param term: tfidf annotation
    :param tf_idf_score: int. tf idf score
    :param tf_idf_score_flag: bool. 
    :return: Bool.
    '''
    
    SofaID=config[ 'Annotation' ].get( 'SOFA_ID' )
    Token_type=config[ 'Annotation' ].get( 'TOKEN_TYPE' )
    
    tf_idf_scores=set( tf_idf_scores )
    
    for other_term in select_covering( cas, SofaID, Token_type, term  ):
        if tf_idf_scores:  #only check if not empty, if empty ==> all should be checked
            if other_term.tfidfValue not in tf_idf_scores: #only interested in covering terms of specific type ( whitelist / obtained via regex )
                continue
        if other_term.begin < term.begin or other_term.end > term.end:
            return False
    return True
    