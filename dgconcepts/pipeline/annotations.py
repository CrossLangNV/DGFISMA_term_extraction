import ahocorasick as ahc
from cassis import Cas, TypeSystem
from typing import List, Tuple, Set, Generator
import re
import string

from .utils import deepest_child, is_token

def add_terms_and_lemmas_to_cas( NLP, cas: Cas, typesystem: TypeSystem, SofaID: str,
                                terms_tf_idf: dict,
                                tagnames: Set[str] = set('p')) -> Cas:
    '''
    Given a cas and its typesystem, this function adds terms and associated tf_idf score (terms_tf_idf) to a given view (SofaID) as type.Tfidf. Annotations will only be added to ValueBetweenTagType elements with tag.tagName in the set tagnames. Returns the same cas object as the input cas, but now with annotations added.
    NLP is the SpaCy model for the extraction of lemmas per token
    '''
    
    Token = typesystem.get_type('de.tudarmstadt.ukp.dkpro.core.api.frequency.tfidf.type.Tfidf')
    Lemma = typesystem.get_type('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma')
       
    A = ahc.Automaton()
    cas_view = cas.get_view(SofaID)
    for term in terms_tf_idf.keys():
        if not term.strip():
            continue
        A.add_word(term, ( terms_tf_idf[term], term ) )
    A.make_automaton()

    for tag in cas_view.select("com.crosslang.uimahtmltotext.uima.type.ValueBetweenTagType"):
        if tag.tagName in set(tagnames):
            
            if not deepest_child(  cas, SofaID, tag, tagnames=tagnames, value_between_tagtype="com.crosslang.uimahtmltotext.uima.type.ValueBetweenTagType" ):
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
    return cas


def add_dependency_annotation( cas:Cas, typesystem: TypeSystem, SofaID:str, defined_terms:Generator[ List[ Tuple[ str, int, int ] ], None, None ], \
                        definition_type:str='de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence', \
                        dependency_type:str="de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency" ):

    definitions=list( cas.get_view( SofaID ).select( definition_type ))
    
    #Sanity check: for each definition, there should be a list of terms (or empty list) that are defined by the definition.
    #This list of terms should either be detected by bert bio tagger, or dependency parser.
    assert len( definitions ) == len( defined_terms )
    
    dpdc = typesystem.get_type( dependency_type )
    
    for defined_terms_sentence, definition in zip( defined_terms, definitions ):
        for defined_term in defined_terms_sentence:
            cas.get_view( SofaID ).add_annotation( dpdc( begin=definition.begin+defined_term[1] , end=definition.begin+defined_term[2], \
                                                                   DependencyType='nsubj' ))
