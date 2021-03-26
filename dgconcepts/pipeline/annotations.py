import ahocorasick as ahc
from cassis import Cas, TypeSystem
from typing import List, Tuple, Set, Generator, Dict
import re
import string

from configparser import ConfigParser

from spacy.lang.en import English      

from .terms_defined.terms_defined_regex import process_definitions_regex
from .utils import deepest_child, is_token, lemmatize

'''
Module contains functions for adding TOKEN_TYPE, LEMMA_TYPE, DEPENDENCY_TYPE and DEFINED_TYPE annotations to a Cas.
'''

def add_token_and_lemma_annotation( NLP: English, cas: Cas, typesystem: TypeSystem, config: ConfigParser ,
                                terms_tf_idf: Dict ):
    '''
    Function adds TOKEN_TYPE annotations and corresponding tf-idf score to a given cas (SOFA_ID view), using a python dictionary (terms_tf_idf). Annotations will only be added to VALUE_BETWEEN_TAG_TYPE elements with tag.tagName in TAG_NAMES. Lemmata of terms are obtained via the NLP spacy model, and added to the Cas.
    
    :param NLP: spacy model. 
    :param cas: Cas. 
    :param typesytem: TypeSystem.
    :param config: ConfigParser.
    :param terms_tf_idf. dict. Keys: terms, Values: tf_idf_score.
    :return: None.     
    '''
    
    SofaID=config[ 'Annotation' ].get( 'SOFA_ID' )
    tagnames=set(config[ 'Annotation' ].get( 'TAG_NAMES' ))
    value_between_tagtype=config[ 'Annotation' ].get( 'VALUE_BETWEEN_TAG_TYPE' )
    token_type=config[ 'Annotation' ].get( 'TOKEN_TYPE' )
    lemma_type=config[ 'Annotation' ].get( 'LEMMA_TYPE' )
    
    Token = typesystem.get_type( token_type )
    Lemma = typesystem.get_type( lemma_type )
    
    #dictionary with terms-lemmas, would otherwise need to lemmatize the same term several times
    terms_lemmas=dict()
        
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
                    if term not in terms_lemmas:
                        terms_lemmas[ term ]=lemmatize( NLP, term )
                    cas_view.add_annotation(
                        Token(begin=tag.begin + start_index, end=tag.begin + end_index + 1, tfidfValue=tfidf,
                              term=term ) )
                    cas_view.add_annotation(
                        Lemma(begin=tag.begin + start_index, end=tag.begin + end_index + 1, value=terms_lemmas[term]))
            
def add_dependency_annotation( cas:Cas, typesystem: TypeSystem, config: ConfigParser, defined_terms:Generator[ List[ Tuple[ str, int, int ] ], None, None ]):
    
    '''
    Function adds DEPENDENCY_TYPE annotations to a Cas. DEPENDENCY_TYPE annotations are only added to positions in the Sofa_string with a DEFINITION_TYPE annotation. The variable defined_terms is a Generator yielding a list of terms and offsets. For each definition (annotated as DEFINITION_TYPE) there is a list of terms and offsets, that are defined by the definition.
    
    :param cas: Cas. 
    :param typesytem: TypeSystem.
    :param config: ConfigParser.
    :param defined_terms. Generator. Yields a list of tuples, with the tuple containing a term, and the begin and end position in the definition.
    :return: None.         
    '''
    
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
    
    '''
    Function adds DEFINED_TERM annotations to a Cas at positions in the Sofa_string with a DEFINITION_TYPE annotation, using DEPENDENCY_TYPE and TOKEN_TYPE annotations already added to the Cas. Depending on the configuration, TOKEN_TYPE with different tf_idf scores will get priority ( whitelisted terms with tf_idf score == TF_IDF_WHITELIST or 'regex' terms with tf_idf_score == TF_IDF_REGEX ).
    
    :param cas: Cas.
    :param typesytem: TypeSystem.
    :param config: ConfigParser.
    :return None.
    '''
    
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
    
    '''
    Function adds DEFINED_TERM annotations to a Cas at positions in the Sofa_string with a DEFINITION_TYPE annotation (sentence_feature), using DEPENDENCY_TYPE and TOKEN_TYPE annotations already added to the Cas. Function only adds TOKEN_TYPE annotation as DEFINED_TERM if it has DEPENDENCY_TYPE overlap, and if it is found via regex (via function process_definitions_regex). 
    If fall_back_regex==True, then, if no DEFINED_TERM is found in the sentence, all terms found via regex will be annotated as DEFINED_TYPE.
    
    :param cas: Cas.
    :param typesytem: TypeSystem.
    :param config: ConfigParser.
    :param sentence_feature: DEFINITION_TYPE feature.
    :param fall_back_regex: bool. If set to True all 
    :return bool.
    '''
    
    Defined_type=typesystem.get_type( config[ 'Annotation' ].get( 'DEFINED_TYPE' ) )
    SofaID= config[ 'Annotation' ].get( 'SOFA_ID' )

    sentence=sentence_feature.get_covered_text()
    
    if sentence:
    
        defined_terms_regex=list(process_definitions_regex(  [  sentence ] , config ))[0]

    else:
        
        return False
        
    defined_terms_regex=[( term[0] , term[1]+sentence_feature.begin, term[2]+sentence_feature.begin ) for term in defined_terms_regex]
    defined_terms_regex=set( defined_terms_regex )
    
    if not defined_terms_regex:
        
        return False
    
    rejected_terms=[]
    defined_term_found=False
    for term in cas.get_view( SofaID ).select_covered( config[ 'Annotation' ].get( 'TOKEN_TYPE' ) ,sentence_feature ):
        
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
    
    '''
    Function adds DEFINED_TERM annotations to a Cas at positions in the Sofa_string with a DEFINITION_TYPE annotation (sentence_feature), using DEPENDENCY_TYPE and TOKEN_TYPE annotations already added to the Cas. Function only adds TOKEN_TYPE annotation as DEFINED_TERM if it has DEPENDENCY_TYPE overlap, and if the TOKEN_TYPE has a tfidfValue present in tf_idf_scores. The longest TOKEN_TYPE will be chosen as DEFINED_TERM in case of overlap.
    If fall_back==True, then, if no DEFINED_TERM is found in the sentence, all TOKEN_TYPE with tfidfValue in tf_idf_scores will be annotated as DEFINED_TYPE. If tf_idf_scores is empty and fall_back==True, then all TOKEN_TYPE's will be annotated.
    
    :param cas: Cas.
    :param typesytem: TypeSystem.
    :param config: ConfigParser.
    :param sentence_feature: DEFINITION_TYPE feature.
    :param fall_back: bool.
    :return bool.
    '''
    
    
    Defined_type = typesystem.get_type( config[ 'Annotation' ].get( 'DEFINED_TYPE' ) )
    SofaID = config[ 'Annotation' ].get( 'SOFA_ID' )
    
    rejected_terms=[]
    defined_term_found=False
    
    for term in cas.get_view( SofaID ).select_covered( config[ 'Annotation' ].get( 'TOKEN_TYPE' ) ,sentence_feature ):
        
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
    
    '''
    Function adds DEFINED_TERM annotations to a Cas at positions in the Sofa_string with a DEFINITION_TYPE annotation (sentence_feature), using DEPENDENCY_TYPE already added to the Cas detected via bio tagging. BIO tagging finds defined_terms, thus we only copy DEPENDENCY_TYPE annotation to a DEFINED_TERM annotation. Function returns defined_terms found in this way. These can then be added as TOKEN_TYPE to the cas.
    
    :param cas: Cas.
    :param typesytem: TypeSystem.
    :param config: ConfigParser.
    :param sentence_feature: DEFINITION_TYPE feature.
    :return List.
    '''


    Defined_type = typesystem.get_type( config[ 'Annotation' ].get( 'DEFINED_TYPE' ) )
    SofaID = config[ 'Annotation' ].get( 'SOFA_ID' )
    
    detected_terms=[]
    
    for dependency in cas.get_view( SofaID ).select_covered( config[ 'Annotation' ].get( 'DEPENDENCY_TYPE' ) ,sentence_feature ):
        
        if not defined_annotation_exists( cas, config, sentence_feature, dependency.begin, dependency.end ):
                        
            detected_terms.append(dependency.get_covered_text().lower())
            
            cas.get_view( SofaID ).add_annotation( Defined_type( begin=dependency.begin, end=dependency.end )  )
            
    return detected_terms #need to annotate these terms in the cas as tfidf
        
        
def has_dependency_annotation( cas, config, term_feature, sentence_feature ):
    
    '''
    Function checks if a TOKEN_TYPE annotation has some overlap with a DEPENDENCY_TYPE annotation in the sentence_feature.
    '''
    
    set1=set(list( range( term_feature.begin, term_feature.end ) ))

    for dependency_feature in cas.get_view( config[ 'Annotation' ][ 'Sofa_ID' ] ).select_covered(( config[ 'Annotation' ][ 'DEPENDENCY_TYPE' ]  ), sentence_feature ):
        set2=set( list( range( dependency_feature.begin, dependency_feature.end ) ) )
        #check if term_feature has some overlapping with a dependency feature ==> if it does, the term is considered to have a dependency annotation
        if set1.intersection( set2 ):
            return True
    
    return False

def defined_annotation_exists( cas:Cas, config:ConfigParser, sentence_feature, begin:int, end:int ):
    
    '''
    Function checks if a DEFINED_TYPE annotation already exists at position begin, end.
    '''
    
    SofaID = config[ 'Annotation' ].get( 'SOFA_ID' )
    
    for defined_type in cas.get_view( SofaID ).select_covered( config[ 'Annotation' ].get( 'DEFINED_TYPE' ), sentence_feature ):
        if defined_type.begin == begin and defined_type.end==end:
            return True
        
    return False
        
def is_longest_term(cas:Cas, config:ConfigParser , term , tf_idf_scores: set() ) -> bool:

    '''
    Function checks if a TOKEN_TYPE annotation (term) is not part of a longer TOKEN_TYPE annotation. 
    Only covering TOKEN_TYPE annotations with specific score (tf_idf_scores) are considered.
    When tf_idf_scores is None, all covering tfidf annotations will be considered to determine if it is the longest term.
    
    :param cas: Cas.
    :param config: ConfigParser. Configuration file.
    :param term: TOKEN_TYPE annotation
    :param tf_idf_score: Set. Tf_idf scores to consider.
    :return: Bool.
    '''
    
    SofaID=config[ 'Annotation' ].get( 'SOFA_ID' )
    Token_type=config[ 'Annotation' ].get( 'TOKEN_TYPE' )
    
    tf_idf_scores=set( tf_idf_scores )
    
    for other_term in cas.get_view( SofaID ).select_covering( Token_type, term  ):
        if tf_idf_scores:  #only check if not empty; if empty ==> all should be checked
            if other_term.tfidfValue not in tf_idf_scores: #only interested in covering terms of specific type ( whitelist / obtained via regex )
                continue
        if other_term.begin < term.begin or other_term.end > term.end:
            return False
    return True
    