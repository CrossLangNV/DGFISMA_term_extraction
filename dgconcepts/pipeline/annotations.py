import ahocorasick as ahc
from cassis import Cas, TypeSystem
from typing import List, Tuple, Set, Generator
import re
import string

from configparser import ConfigParser

from .terms_defined_regex import process_definitions_regex
from .utils import deepest_child, is_token

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
    
    #Sanity check: for each definition, there should be a list of terms (or empty list) that are defined by the definition.
    #This list of terms should either be detected by bert bio tagger, or dependency parser.
    assert len( definitions ) == len( defined_terms )
    
    dpdc = typesystem.get_type( dependency_type )
    
    for defined_terms_sentence, definition in zip( defined_terms, definitions ):
        for defined_term in defined_terms_sentence:
            cas.get_view( SofaID ).add_annotation( dpdc( begin=definition.begin+defined_term[1] , end=definition.begin+defined_term[2], \
                                                                   DependencyType='nsubj' ))
            
            
def add_defined_term_regex( cas, typesystem , config , sentence_feature, fall_back_regex=True ):
    
    Defined_type=typesystem.get_type( config[ 'Annotation' ].get( 'DEFINED_TYPE' ) )
    SofaID= config[ 'Annotation' ].get( 'SOFA_ID' )

    defined_terms_regex=list(process_definitions_regex(  [  sentence_feature.get_covered_text() ] ))[0]
    defined_terms_regex=[( term[0] , term[1]+sentence_feature.begin, term[2]+sentence_feature.begin ) for term in defined_terms_regex]
    
    rejected_terms=[]
    defined_term_found=False
    for term in cas.get_view( SofaID ).select_covered( config[ 'Annotation' ].get( 'TOKEN_TYPE' ) , sentence_feature ):
        
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
    
    for term in cas.get_view( SofaID ).select_covered( config[ 'Annotation' ].get( 'TOKEN_TYPE' ) , sentence_feature ):
        
        if tf_idf_scores and term.tfidfValue not in tf_idf_scores: #only interested in terms with specific scores ( whitelisted and regex detected terms )
            continue
        
        #check if it is the 'longest' whitelisted term at that location...
        if is_longest_term( cas, config, term, tf_idf_scores ): # and not defined_annotation_exists( cas,config,term.begin,term.end ):

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

def add_defined_term_bert():
    
    bert=True
    
    if bert:
        pass
        #and fallback...
        
        
#helper functions
def has_dependency_annotation( cas, config, term_feature, sentence_feature ):
    
    set1=set(list( range( term_feature.begin, term_feature.end ) ))

    for dependency_feature in cas.get_view( config[ 'Annotation' ][ 'Sofa_ID' ] ).select( config[ 'Annotation' ][ 'DEPENDENCY_TYPE' ]  ):
        set2=set( list( range( dependency_feature.begin, dependency_feature.end ) ) )
        #check if term_feature has some overlapping with a dependency feature
        if set1.intersection( set2 ):
            return True
    
    return False

def defined_annotation_exists( cas:Cas, config:ConfigParser, begin:int, end:int ):
    
    for defined_type in cas.get_view( config[ 'Annotation' ]['Sofa_ID']  ).select( config[ 'Annotation' ].get( 'DEFINED_TYPE' ) ):
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
    
    for other_term in cas.get_view(  SofaID ).select_covering( Token_type  , term ):
        if tf_idf_scores:  #only check if not empty, if empty ==> all should be checked
            if other_term.tfidfValue not in tf_idf_scores: #only interested in covering terms of specific type ( whitelist / obtained via regex )
                continue
        if other_term.begin < term.begin or other_term.end > term.end:
            return False
    return True  
    
    
'''
        
def add_defined_term_dependency_parser( cas: Cas, typesystem: TypeSystem, config: ConfigParser ):
    
    SofaID=config[ 'Annotation' ].get( 'SOFA_ID' )

    token_type=config[ 'Annotation' ].get( 'TOKEN_TYPE' )
    definition_type=config['Annotation'].get( 'DEFINITION_TYPE' )
    defined_type=config[ 'Annotation' ].get( 'DEFINED_TYPE' )
    dependency_type=config[ 'Annotation' ].get( 'DEPENDENCY_TYPE' )
        
    tf_idf_whitelist=config[ 'TermExtraction' ].getfloat( 'TFIDF_WHITELIST' )
    tf_idf_regex=config[ 'TermExtraction' ].getfloat( 'TFIDF_REGEX' )
    tf_idf_threshold=config[ 'TermExtraction' ].getfloat( 'TFIDF_THRESHOLD' )
    
    prioritize_regex=config[ 'DefinedTerm' ].getboolean( 'PRIORITIZE_REGEX' )
    prioritize_whitelist=config[ 'DefinedTerm' ].getboolean( 'PRIORITIZE_WHITELIST' )
    fallback_to_regex=config[ 'DefinedTerm' ].getboolean( 'FALLBACK_TO_REGEX' )
    fallback_to_whitelist=config[ 'DefinedTerm' ].getboolean( 'FALLBACK_TO_WHITELIST' )
                
    Token = typesystem.get_type( defined_type )
    
    #iteration over the definitions
    for definition in cas.get_view( SofaID ).select( definition_type ):
    
        defined_detected=False
        
        terms_sentence = list(cas.get_view( SofaID ).select_covered( token_type, definition  ) )
        
        if prioritize_regex:
            
            for tf_idf in terms_sentence:
                #case where tf_idf term is found via regex and has a good dependency type
                if (tf_idf.tfidfValue == tf_idf_regex and defined_term(cas, config, tf_idf )=='nsubj' ): #make this configurable
                    if is_longest_term( cas, config, tf_idf, tf_idf_regex, True ): #check if there is a longer covering terms with same tf_idf_score
                        cas.get_view( SofaID ).add_annotation( Token( begin=tf_idf.begin , end=tf_idf.end ) )
                        defined_detected=True

            #if one of the terms found via regex is considered the term confirmed via dependency parser, stop searching.
            if defined_detected:
                continue

            if fallback_to_regex:
                for tf_idf in terms_sentence:
                    #case where tf_idf term is found via regex and does not have a bad dependency type
                    if (tf_idf.tfidfValue == tf_idf_regex and defined_term(cas, config, tf_idf )!='bad' ): 
                        if is_longest_term( cas, config, tf_idf, tf_idf_regex, True ): #check if there is a longer covering term with same tf_idf_score (via regex)
                            cas.get_view( SofaID ).add_annotation( Token( begin=tf_idf.begin , end=tf_idf.end ) )
                            defined_detected=True   
                        
        #if one of the terms found via regex is not considered bad, stop searching
        if defined_detected:
            continue
          
        if prioritize_whitelist:
        
            for tf_idf in terms_sentence:
                #case where tf_idf term is whitelisted and has a good dependency type
                if (tf_idf.tfidfValue == tf_idf_whitelist and defined_term(cas, config, tf_idf)=='nsubj' ): #make this configurable
                    if is_longest_term( cas, config, tf_idf, tf_idf_whitelist, True ): #check if there is a longer covering term with same tf_idf_score (whitelisted)
                        cas.get_view( SofaID ).add_annotation( Token( begin=tf_idf.begin , end=tf_idf.end ) )
                        defined_detected=True
                        
            if defined_detected:
                continue
                
            if fallback_to_whitelist:
                for tf_idf in terms_sentence:
                    #case where tf_idf term is whitelisted and has a good dependency type
                    if (tf_idf.tfidfValue == tf_idf_whitelist and defined_term(cas, config, tf_idf )!='bad' ): #make this configurable
                        if is_longest_term( cas, config, tf_idf, tf_idf_whitelist, True ): #check if there is a longer covering term with same tf_idf_score (whitelisted)
                            cas.get_view( SofaID ).add_annotation( Token( begin=tf_idf.begin , end=tf_idf.end ) )
                            defined_detected=True
                
        #if one of the whitelisted terms is considered the defined term, stop searching
        if defined_detected:
            continue
            
        #case when there are not whitelisted/regex terms in the sentence, and/or all whitelisted terms were rejected by dependency parser 
        #for bert bio_tagger you probably want fallback to bio-tagged item...
        for tf_idf in terms_sentence:
            if defined_term(cas, config, tf_idf ) =='nsubj' and tf_idf.tfidfValue > tf_idf_threshold:
                if is_longest_term( cas, config, tf_idf, 0 , False ): #check if there is a longer covering term (with any tf_idf_score)
                    cas.get_view( SofaID ).add_annotation( Token( begin=tf_idf.begin  , end=tf_idf.end ) )
                    defined_detected=True


def defined_term( cas:Cas, config:ConfigParser, term )-> str:

    Function to find dependency relation of a tfidf annotation (term).

    :param cas: Cas.
    :param config: ConfigParser. Configuration file.
    :param term: tfidf annotation
    :return: String.

    SofaID=config[ 'Annotation' ].get( 'SOFA_ID' )
    dependency_type=config[ 'Annotation' ].get( 'DEPENDENCY_TYPE' )
    
    for dependency in cas.get_view( SofaID ).select_covered( dependency_type, term ):
        if dependency.DependencyType == 'bad':
            return 'bad'
        elif dependency.DependencyType == 'nsubj':
            return 'nsubj'

    return 'no_dependency_found'
    
'''