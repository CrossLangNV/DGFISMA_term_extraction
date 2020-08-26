from .utils import find_indices_term
from typing import List, Tuple, Set

from cassis import Cas, TypeSystem


def add_terms_to_cas( cas: Cas, typesystem: TypeSystem, SofaID: str, terms_tf_idf: List[Tuple[ str, float ]] , tagnames: Set[str] = set( 'p' )   ) -> Cas:

    '''
    Given a cas and its typesystem, this function adds terms and associated tf_idf score (terms_tf_idf) to a given view (SofaID) as type.Tfidf. Annotations will only be added to ValueBetweenTagType elements with tag.tagName in the set tagnames. Returns the same cas object as the input cas, but now with annotations added.
    '''
    
    Token = typesystem.get_type( 'de.tudarmstadt.ukp.dkpro.core.api.frequency.tfidf.type.Tfidf' ) 

    cas_view=cas.get_view(  SofaID )

    for tag in cas_view.select( "com.crosslang.uimahtmltotext.uima.type.ValueBetweenTagType" ):

        if tag.tagName in set(tagnames):
            for term, tfidf in terms_tf_idf:
                matches=list(find_indices_term( term, tag.get_covered_text() ))
                if not matches:
                    continue
                for match in matches:
                    cas_view.add_annotation( Token(begin=tag.begin+match.start() , end=tag.begin+match.end(), tfidfValue=tfidf, term=term ) )
                    
    return cas