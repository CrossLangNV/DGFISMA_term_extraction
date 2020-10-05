import re
import string
from collections import Counter

### Might use for normalisation
def find_indices_term(term: str, sentence: str):
    '''
    Find matches of a given term in a given sentence using a regex. Different variants of the term will also be matches. Returns a Callable.
    '''

    # first translate terms like "regulation-offices" to "regulation offices"
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))  # map punctuation to space
    term = term.translate(translator)  # replace punctuation in the term with ' '

    term = term.replace(" ", "")  # remove all spaces

    words = "([- ]*)".join([char for char in
                            term])  # will match different variants. I.e. if term is "regulation offices" ,
    # regex also matches regulation-offices and regulationoffices
    term_regex = r"\b" + f"({words})" + r"\b"

    return re.finditer(term_regex, sentence, re.IGNORECASE)

def flatten_term_dicts(all_terms_n_tf):
    """
    :param all_terms_n_tf: a list of dictionaries with ngrams, counts and tf
    :return: a flat dictionary with unique ngrams, counts and tf
    """
    flattened_term_dict = {'ngrams', 'count', 'tf'}
    print(flattened_term_dict)

    return flattened_term_dict