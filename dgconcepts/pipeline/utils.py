import sys
from typing import List, Tuple, Set

from cassis import Cas

def get_sentences(  cas: Cas, SofaID: str , tagnames : Set[str] = set( 'p'), \
                  value_between_tagtype="com.crosslang.uimahtmltotext.uima.type.ValueBetweenTagType"  ) -> (List[str], List[Tuple[ int,int ]]):
    
    '''
    Given a cas, and a view (SofaID), this function selects all ValueBetweenTagType elements ( with tag.tagName in the set tagnames ), extracts the covered text, and returns the list of extracted sentences and a list of Tuples containing begin and end posistion of the extracted sentence in the sofa.
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