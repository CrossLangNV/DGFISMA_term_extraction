import re
import string


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


def beautiprint(title,list, toFile=None):
    """ Print out a list with a title """
    print(title, file=toFile)
    print("#######################", file=toFile)
    print("", file=toFile)
    for l in list:        
        print(l, file=toFile)
    print("", file=toFile)   