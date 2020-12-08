from typing import Tuple, List, Dict
from re import Pattern

import time
import re
import string

from .utils import get_sentences
from .terms import get_terms, remove_add_update_terms_blacklist_whitelist

from .terms_defined.terms_defined_dependency_parser import process_definitions_dependency_parser
from .terms_defined.terms_defined_bio_tagging import process_definitions_bert_bio_tagging, TrainedBertBIOTagger
from .terms_defined.terms_defined_regex import process_definitions_regex

from .annotations import add_token_and_lemma_annotation, add_dependency_annotation, add_defined_term_annotation

from spacy.lang.en import English

from cassis import Cas, TypeSystem
from configparser import ConfigParser

def concept_extraction( NLP: English, trained_bert_bio_tagger: TrainedBertBIOTagger, cas: Cas, typesystem: TypeSystem, config: ConfigParser, \
                       whitelist_blacklist: Tuple[ List[str], List[str] ]=( [],[] )):

    '''
    Function to perform term extraction (using spacy model NLP), and detection of the defined terms (using a spacy dependency parser or a TrainedBertBIOTagger, depending on the configuration). Terms and defined terms will be annotated in the Cas.
    
    :param NLP: spacy model. 
    :param trained_bert_bio_tagger: TrainedBertBIOTagger. 
    :param cas: Cas. 
    :param typesytem: TypeSystem.
    :param config: ConfigParser
    :param whitelist_blacklist. Contains two lists: one of whitelisted terms, and one of blacklisted terms.
    :return: None.        
    '''
    
    sentences, _ = get_sentences( cas, SofaID=config[ 'Annotation' ].get( 'SOFA_ID' ), tagnames=set(config[ 'Annotation' ].get( 'TAG_NAMES' )), \
                                 value_between_tagtype=config[ 'Annotation' ].get( 'VALUE_BETWEEN_TAG_TYPE' )   )

    print( "Start term extraction." )
    start=time.time()

    #get a dictionary with all detected terms and tfidf scores
    terms_n_tfidf, _ = get_terms( NLP, sentences, \
                                 extract_supergrams = config[ 'TermExtraction' ].getboolean( 'EXTRACT_SUPERGRAMS' ), \
                                 nMax = config[ 'TermExtraction' ].getint( 'MAX_LEN_NGRAM' ) )

    print( f"Term extraction took { time.time() -start } seconds." )

    #get the definitions (previously annotated via definition detector)
    definitions=[definition.get_covered_text() for definition in \
    cas.get_view( config[ 'Annotation' ].get( 'SOFA_ID' ) ).select( config[ 'Annotation' ].get( 'DEFINITION_TYPE' )  ) ]

    #get a list of terms defined in detected definitions, via a regex (i.e. in between quotes)
    terms_regex=process_definitions_regex( definitions, config ) 
    terms_regex=list( terms_regex )

    #sanity check
    assert len(terms_regex) == len( definitions )

    terms_defined=[]
    for terms_sentence in terms_regex:
        terms_defined+=[term_sentence[0] for term_sentence in terms_sentence  ]
    terms_defined=set(terms_defined)

    for term in terms_defined:
        terms_n_tfidf.update( { term: config[ 'TermExtraction' ].getfloat( 'TFIDF_REGEX' ) }  )

    #remove terms found in definitions between quotes from the whitelist (because these are already in dictionary terms_n_tfidf)
    whitelist=set( whitelist_blacklist[0] ) - set( terms_defined )
    blacklist=set( whitelist_blacklist[1] )

    #add/remove terms from whitelist, blacklist
    terms_n_tfidf = remove_add_update_terms_blacklist_whitelist( terms_n_tfidf, whitelist, blacklist, \
                                                                tf_idf_whitelist=config[ 'TermExtraction' ].getfloat( 'TFIDF_WHITELIST' ) )
    
    print( "Start annotation terms and lemmas." )
    start=time.time()
    #annotate tokens and lemmas in the cas (lemmatization done via Spacy model)
    #first remove terms consisting only of punctuation/numbers
    clean_dictionary( terms_n_tfidf  )
    add_token_and_lemma_annotation( NLP, cas, typesystem, config, terms_n_tfidf ) 
    print( f"Annotation of terms and lemmas took { time.time() -start } seconds." )
    
    #obtain dependency annotation (what is defined)
    if config[ 'DefinedTerm' ].getboolean( 'BERT_BIO_TAGGING' ):
        terms_dependency=process_definitions_bert_bio_tagging( definitions, trained_bert_bio_tagger,\
                                                              gpu=config[ 'BertBIOTagger' ].getint( 'GPU' ),\
                                                              seq_length=config[ 'BertBIOTagger' ].getint( 'SEQ_LENGTH' ),\
                                                              batch_size=config[ 'BertBIOTagger' ].getint( 'BATCH_SIZE' ) )
    else:
        terms_dependency=process_definitions_dependency_parser( definitions, NLP )

    #add the depedency annotation
    add_dependency_annotation( cas, typesystem, config, terms_dependency )

    #use the dependency annotation to determine which term is defined
    terms=add_defined_term_annotation( cas, typesystem, config )

    # if BERT BIO tagger detected a term that was not found previously ==> annotate as tf-idf feature
    # if dependency parser is used instead of BERT BIO tagger, this list of terms will always be empty
    if terms:  
        terms_bert={}
        terms=list(set( terms ))
        
        for term in terms:
            if term not in terms_n_tfidf and term not in blacklist:
                terms_bert[ term  ]=config[ 'TermExtraction' ].getfloat( 'TFIDF_BERT' )
        clean_dictionary( terms_bert  )
        if terms_bert:
            add_token_and_lemma_annotation( NLP, cas, typesystem, config, terms_bert ) 
            
#helper function to clean dictionary of terms
def clean_dictionary( terms_dict: Dict )-> None:
    
    '''
    Helper function to remove invalid terms from a dictionary using regex.
    :param terms_dict: Dict. 
    :return: None.
    '''
    
    extra_chars_to_exclude=[ '∙', ' ', "—", "…", "·", "+", "“", "√" , "≤", "<",'≥', ">" ,"⋅", "■", "£", "½", "÷" ]  
    
    #exclude terms matching this pattern ( i.e. terms consisting only of numericals / punctuation )
    punctuation_chars=string.punctuation+"".join( extra_chars_to_exclude)
    pattern = re.compile("[0-9{}]+$".format(re.escape( punctuation_chars )))
    #r'[\d!"\#\$%\&\'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~∙\ —…·\+“√≤<≥>⋅■£½÷]+$'

    keys=set( terms_dict.keys() )
    for term in keys:
        if pattern.match( term ):
            terms_dict.pop( term )