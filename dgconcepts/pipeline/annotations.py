import ahocorasick as ahc
from cassis import Cas, TypeSystem
from typing import List, Tuple, Set
import re
import string

from .utils import deepest_child

def proceed_with_annotation(start_index, end_index, text):
    
    #trivial case (no text)
    if not text:
        return False
    
    #trivial case (start_index equal to end_index)
    elif start_index==end_index:
        return False
    
    #e.g. 'livestock' in 'livestock'
    elif start_index == 0 and end_index == len( text ) - 1:
        return True
    
    #e.g. 'livestock' in 'livestock some'
    elif start_index == 0:
        if text[ end_index+1 ].isalpha():
            return False
        
    #e.g. 'livestock' in 'some livestock'
    elif end_index == len( text ) -1:
        if text[start_index -1].isalpha():
            return False
        
    #e.g. 'livestock' in 'some livestock some'      
    else:
        if text[ start_index-1 ].isalpha() or text[end_index+1].isalpha():
            return False
        
    return True


def add_terms_and_lemmas_to_cas(NLP, cas: Cas, typesystem: TypeSystem, SofaID: str,
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
                if proceed_with_annotation(start_index, end_index, text):
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
