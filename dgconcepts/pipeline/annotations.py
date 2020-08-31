from typing import List, Tuple, Set

from cassis import Cas, TypeSystem
import ahocorasick as ahc


def add_terms_to_cas(cas: Cas, typesystem: TypeSystem, SofaID: str, terms_tf_idf: List[Tuple[str, float]],
                     tagnames: Set[str] = set('p')) -> Cas:
    '''
    Given a cas and its typesystem, this function adds terms and associated tf_idf score (terms_tf_idf) to a given view (SofaID) as type.Tfidf. Annotations will only be added to ValueBetweenTagType elements with tag.tagName in the set tagnames. Returns the same cas object as the input cas, but now with annotations added.
    '''

    Token = typesystem.get_type('de.tudarmstadt.ukp.dkpro.core.api.frequency.tfidf.type.Tfidf')
    A = ahc.Automaton()
    cas_view = cas.get_view(SofaID)
    for (key, cat) in terms_tf_idf:
        A.add_word(key.strip(), (cat, key.strip()))
    A.make_automaton()

    for tag in cas_view.select("com.crosslang.uimahtmltotext.uima.type.ValueBetweenTagType"):
        if tag.tagName in set(tagnames):
            text = tag.get_covered_text()
            try:
                for end_index, (tfidf, term) in A.iter(text):
                    start_index = end_index - len(term) + 1
                    cas_view.add_annotation(
                        Token(begin=tag.begin + start_index, end=tag.begin + end_index + 1, tfidfValue=tfidf, term=term))
            except:
                continue

    return cas