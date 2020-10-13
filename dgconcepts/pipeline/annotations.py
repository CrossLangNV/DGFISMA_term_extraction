import ahocorasick as ahc
from cassis import Cas, TypeSystem
from typing import List, Tuple, Set

def add_terms_and_lemmas_to_cas(NLP, cas: Cas, typesystem: TypeSystem, SofaID: str, terms_tf_idf: List[Tuple[str, float]],
                     tagnames: Set[str] = set('p')) -> Cas:
    '''
    Given a cas and its typesystem, this function adds terms and associated tf_idf score (terms_tf_idf) to a given view (SofaID) as type.Tfidf. Annotations will only be added to ValueBetweenTagType elements with tag.tagName in the set tagnames. Returns the same cas object as the input cas, but now with annotations added.
    NLP is the SpaCy model for the extraction of lemmas per token
    '''
    Token = typesystem.get_type('de.tudarmstadt.ukp.dkpro.core.api.frequency.tfidf.type.Tfidf')
    Lemma = typesystem.get_type('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma')
    
    A = ahc.Automaton()
    cas_view = cas.get_view(SofaID)
    for (key, cat) in terms_tf_idf:
        A.add_word(key.strip(), (cat, key.strip()))
    A.make_automaton()

    for tag in cas_view.select("com.crosslang.uimahtmltotext.uima.type.ValueBetweenTagType"):
        conditionMet = False
        if tag.tagName in set(tagnames):
            if all(x.tagName in set(tagnames) for x in list(cas.get_view(SofaID).select_covered("com.crosslang.uimahtmltotext.uima.type.ValueBetweenTagType", tag))):
                    conditionMet = True
        if conditionMet:
            text = tag.get_covered_text().lower()
            try:
                for end_index, (tfidf, term) in A.iter(text):
                    start_index = end_index - len(term) + 1
                    proceed_with_annotation = False

                    # annotate if term is the first word in text, and term is not followed by a letter,
                    # e.g. 'livestock' in 'livestock units'
                    if start_index == end_index - len(term)-1 and not text[end_index + 1].isalpha():
                        proceed_with_annotation = True

                    # annotate if term is the last word in text, and term is not preceded by a letter, e.g. 'unit' in
                    # 'livestock unit'
                    if end_index == len(text)-1 and not text[start_index - 1].isalpha():
                        proceed_with_annotation = True

                    # annotate if term is somewhere in text, and term is neither preceded nor followed by a letter,
                    # e.g. 'stock' in 'live stock unit'
                    if not text[start_index-1].isalpha() and start_index != end_index - len(term)-1 and not text[end_index+1].isalpha() and end_index != len(text)-1:
                        proceed_with_annotation = True

                    if proceed_with_annotation:
                        lemmas = []
                        for word in NLP(term):
                            lemmas.append(word.lemma_)
                        term_lemmas = ' '.join(lemmas)
                        cas_view.add_annotation(
                            Token(begin=tag.begin + start_index, end=tag.begin + end_index + 1, tfidfValue=tfidf, term=term))
                        cas_view.add_annotation(Lemma(begin=tag.begin + start_index, end=tag.begin + end_index + 1, value=term_lemmas))
            except:
                continue
    return cas
