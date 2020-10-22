import ahocorasick as ahc
from cassis import Cas, TypeSystem
from typing import List, Tuple, Set
import string
import re


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
    return cas

def add_checked_term_to_cas(cas_view, sentence, np, TYPESYSTEM):
    A = ahc.Automaton()
    Token = TYPESYSTEM.get_type('de.tudarmstadt.ukp.dkpro.core.api.frequency.tfidf.type.Tfidf')
    A.add_word(np.text, (1, np.text.strip()))
    A.make_automaton()
    for end_index, (tfidf, term) in A.iter(sentence.get_covered_text()):
        start_index = end_index - len(term) + 1
        cas_view.add_annotation(
            Token(begin=sentence.begin + start_index, end=sentence.begin + end_index + 1, tfidfValue=-1,
                  term=np.text))

def check_if_term_annotated(cas_view, sentence, np):
    terms = [token.get_covered_text() for token in
             cas_view.select_covered("de.tudarmstadt.ukp.dkpro.core.api.frequency.tfidf.type.Tfidf", sentence)]
    if np.text not in terms:
        return False
    else:
        return True

def check_definitions(cas, NLP, TYPESYSTEM):
    cas_view = cas.get_view("html2textView")
    for sentence in cas_view.select("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"):
        definition = sentence.get_covered_text()
        definition = re.sub(r'\(\w{1}\)', '', definition)
        definition = definition.translate(str.maketrans('', '', string.punctuation))
        definition = definition.strip()
        doc = NLP(definition)
        for np in doc.noun_chunks:
            for token in np:
                if token.dep_ == 'nsubj' and token.head.dep_ == 'ROOT':
                    term_has_been_annotated = check_if_term_annotated(cas_view, sentence, np)
                    if term_has_been_annotated:
                        continue
                    else:
                        add_checked_term_to_cas(cas_view, sentence, np, TYPESYSTEM)
    return cas