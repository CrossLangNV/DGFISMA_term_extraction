import ahocorasick as ahc
from cassis import Cas, TypeSystem
from typing import List, Tuple, Set
import re
import string

TFIDF_FOR_WHITELISTED_SUBJECTS = -2
TFIDF_FOR_ANNOTATED_SUBJECTS = -1


def proceed_with_annotation(start_index, end_index, term, text):
    proceed_with_annotation = False

    # annotate if term is the first word in text, and term is not followed by a letter,
    # e.g. 'livestock' in 'livestock units'
    if start_index == end_index - len(term) - 1 and not text[end_index + 1].isalpha():
        proceed_with_annotation = True

    # annotate if term is the last word in text, and term is not preceded by a letter, e.g. 'unit' in
    # 'livestock unit'
    if end_index == len(text) - 1 and not text[start_index - 1].isalpha():
        proceed_with_annotation = True

    # annotate if term is somewhere in text, and term is neither preceded nor followed by a letter,
    # e.g. 'stock' in 'live stock unit'
    if not text[start_index - 1].isalpha() and start_index != end_index - len(term) - 1 and not text[
        end_index + 1].isalpha() and end_index != len(text) - 1:
        proceed_with_annotation = True

    return proceed_with_annotation


def add_terms_and_lemmas_to_cas(NLP, cas: Cas, typesystem: TypeSystem, SofaID: str,
                                terms_tf_idf: List[Tuple[str, float]],
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
            if all(x.tagName in set(tagnames) for x in list(
                    cas.get_view(SofaID).select_covered("com.crosslang.uimahtmltotext.uima.type.ValueBetweenTagType",
                                                        tag))):
                conditionMet = True
        if conditionMet:
            text = tag.get_covered_text().lower()
            for end_index, (tfidf, term) in A.iter(text):
                start_index = end_index - len(term) + 1
                if proceed_with_annotation(start_index, end_index, term, text):
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


def add_checked_term_to_cas(cas_view, sentence, np, TYPESYSTEM, tfidf_score):
    A = ahc.Automaton()
    Token = TYPESYSTEM.get_type('de.tudarmstadt.ukp.dkpro.core.api.frequency.tfidf.type.Tfidf')
    A.add_word(np.text, (1, np.text.strip()))
    A.make_automaton()
    for end_index, (tfidf, term) in A.iter(sentence.get_covered_text()):
        start_index = end_index - len(term) + 1
        cas_view.add_annotation(
            Token(begin=sentence.begin + start_index, end=sentence.begin + end_index + 1, tfidfValue=tfidf_score,
                  term=np.text))


def get_definition_subject(NLP, sentence):
    """

    :param NLP: the SpaCy model
    :param sentence: object of type "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
    :return: spacy.tokens.span.Span
    """
    definition = sentence.get_covered_text()
    definition = re.sub(r'\(\w{1}\)', '', definition)  # remove (a) or (b) before the definition
    definition = re.sub(r'\(\d{1, 3}\.\)', '', definition)  # remove 1. or 2. before the definition
    definition = definition.translate(str.maketrans('', '', string.punctuation.replace('-', '').replace('/', '')))

    definition = definition.strip()
    doc = NLP(definition)
    definition_subject = ''
    for np in doc.noun_chunks:
        for token in np:
            if token.dep_ == 'nsubj' and token.head.dep_ == 'ROOT':
                definition_subject = np

    return definition_subject


def check_definitions(cas, NLP, TYPESYSTEM):
    cas_view = cas.get_view("html2textView")
    for sentence in cas_view.select("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"):
        definition_subject = get_definition_subject(NLP, sentence)
        token_annotations = cas_view.select_covered('de.tudarmstadt.ukp.dkpro.core.api.frequency.tfidf.type.Tfidf',
                                                  sentence)
        annotated_terms = [annotation.term for annotation in token_annotations]

        no_terms_were_detected = False
        subject_was_not_detected = False

        # if no terms were annotated
        if len(annotated_terms) == 0:
            no_terms_were_detected = True
            add_checked_term_to_cas(cas_view, sentence, definition_subject, TYPESYSTEM, TFIDF_FOR_ANNOTATED_SUBJECTS)

        #  subject is not a whitelisted term and has not been annotated
        if definition_subject.text not in annotated_terms:
            subject_was_not_detected = True
            add_checked_term_to_cas(cas_view, sentence, definition_subject, TYPESYSTEM, TFIDF_FOR_ANNOTATED_SUBJECTS)

        if (no_terms_were_detected and subject_was_not_detected) == False:
            for token in token_annotations:
                #  subject is a whitelisted term
                if token.tfidfValue == TFIDF_FOR_ANNOTATED_SUBJECTS and token.term == definition_subject.text:
                    add_checked_term_to_cas(cas_view, sentence, definition_subject, TYPESYSTEM,
                                            TFIDF_FOR_WHITELISTED_SUBJECTS)

                # subject is not a whitelisted term and has been annotated
                if token.tfidfValue != (
                        TFIDF_FOR_ANNOTATED_SUBJECTS or TFIDF_FOR_WHITELISTED_SUBJECTS) and token.term == definition_subject.text:  # sub
                    add_checked_term_to_cas(cas_view, sentence, definition_subject, TYPESYSTEM,
                                            TFIDF_FOR_ANNOTATED_SUBJECTS)

    return cas
