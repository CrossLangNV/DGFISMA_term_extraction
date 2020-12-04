import sys
from typing import List, Tuple, Set

from cassis import Cas

from typing import Generator

from spacy.lang.en import English      

def get_sentences(  cas: Cas, SofaID: str , tagnames : Set[str] = set( 'p'), \
                  value_between_tagtype="com.crosslang.uimahtmltotext.uima.type.ValueBetweenTagType"  ) -> (List[str], List[Tuple[ int,int ]]):
    
    '''
    Given a cas, and a view (SofaID), this function selects all VALUE_BETWEEN_TAG_TYPE elements ( with tag.tagName in TAG_NAMES ), extracts the covered text, and returns the list of extracted sentences and a list of Tuples containing begin and end posistion of the extracted sentence in the sofa.
    Function will only extract text of the deepest child of the to be extracted tagnames.
        
    :param cas: cassis.typesystem.Typesystem. Corresponding Typesystem of the cas.
    :param SofaID: String. Name of the sofa.
    :param tagnames: String. tagtypes to extract.
    :return: Tuple. Tuple with extracted text and the begin and end postion of the extracted text in the sofa.
    '''
            
    sentences=[]
    begin_end_position=[]
    for tag in cas.get_view( SofaID ).select( value_between_tagtype):
        if tag.tagName in set(tagnames) and deepest_child(  cas, SofaID, tag, tagnames=tagnames, value_between_tagtype=value_between_tagtype ):
            sentence=tag.get_covered_text().strip()
            if sentence:
                sentences.append( sentence )
                begin_end_position.append( ( tag.begin, tag.end )  )
                
    return sentences, begin_end_position

#helper function to check if a tag is nested or not
def deepest_child( cas:Cas, SofaID:str , tag ,tagnames: Set[str] = set( 'p' ), \
                  value_between_tagtype= "com.crosslang.uimahtmltotext.uima.type.ValueBetweenTagType" ) -> bool:

    if len( [item for item in cas.get_view( SofaID ).select_covered(  value_between_tagtype , tag ) \
             if (item.tagName in tagnames and item.get_covered_text() ) ] ) > 1:
        return False
    else:
        return True
    
    
def is_token(start_index:int, end_index:int, text:str, special_characters:List[str]=[ "-","_","+"]) -> bool:
    
    '''
    Given a start index, and end_index, and a string, this function checks if token(s) covered by the span is part of other token(s).
    
    :param start_index: int.
    :param end_index: int. 
    :param special_characters: List. List of special characters treated as alpha characters
    :return: bool.
    '''
    
    if start_index <0 or end_index<0:
        raise ValueError(f"Both start_index and end_index should be >0, however start_index is {start_index} and end_index is {end_index}")

    elif end_index<start_index:
        raise ValueError(f"end_index should be > start_index, however start_index is {start_index} and end_index is {end_index}")
        
    elif end_index > (len( text ) -1) :   #this one also takes care of no text case
        raise ValueError(f"end_index should be < len(text) -1, however end_index is {end_index} and len(text) is {len(text)}")
 
    #set of special characters treated as alpha characters
    #e.g.: the term 'livestock' in 'some livestock-some some' should not be annotated, but 'livestock' in 'some "livestock" some' should.
    special_characters=set(special_characters)
        
    #trivial case (start_index equal to end_index)
    #if start_index==end_index:
        #return False
    
    #e.g. 'livestock' in 'livestock'
    if start_index == 0 and end_index == len( text ) - 1:
        return True
    
    #e.g. 'livestock' in 'livestock some'
    elif start_index == 0:
        if (text[ end_index+1 ].isalpha() or text[ end_index+1 ] in special_characters ):
            return False
        
    #e.g. 'livestock' in 'some livestock'
    elif end_index == len( text ) -1:
        if (text[start_index -1].isalpha() or text[start_index -1] in special_characters ):
            return False
        
    #e.g. 'livestock' in 'some livestock some';      
    else:
        if (text[ start_index-1 ].isalpha() or text[ start_index-1 ] in special_characters ) \
        or (text[end_index+1].isalpha() or text[end_index+1] in special_characters ):
            return False
        
    return True

def lemmatize( NLP: English, term:str )->str:
    '''
    Lemmatize (multi-word) term using a spacy model.
    
    :param NLP: English.
    :param term: string. 
    :return: string.
    '''
    
    lemma = []
    for word in NLP(term):
        lemma.append(word.lemma_)
    term_lemma = ' '.join(lemma)
    return term_lemma

