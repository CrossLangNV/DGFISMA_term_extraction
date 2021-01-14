import warnings
from pathlib import Path

import plac

from similar_terms.methods import SimilarWordsRetriever


@plac.annotations(
    path_term=("Path of text-file with single term.",),
    path_terms_voc=("Path of text-file with terms separated over each line.",),
    path_out=("Path to export to.",)
)
def main(
        path_term: Path,
        path_terms_voc: Path,
        path_out: Path
):
    """

    Args:
        path_term: File, containing single term
        path_terms_voc: File, containing vocabulary with a term on each row.
        path_out: File to where to export the similar terms.

    Returns: list with terms from the vocabulary similar to the provided term.
    """

    warnings.warn("Deprecated user-script", DeprecationWarning)

    path_term = Path(path_term)
    path_terms_voc = Path(path_terms_voc)

    with open(path_term) as f:
        term = f.read().splitlines()[0]

    with open(path_terms_voc) as f:
        terms_voc = f.read().splitlines()

    similar_words_retriever = SimilarWordsRetriever(terms_voc)

    similar_terms = similar_words_retriever.get_similar_foo(term, terms_voc)

    similar_terms2 = similar_words_retriever.get_similar_k(term, 5)

    with open(path_out, 'w+') as f:
        # assert term ~= similar_terms[0]

        for sim_term, index, score in zip(*similar_terms[1]):
            f.write('; '.join(map(str, (sim_term, index, score))) + '\n')

    return similar_terms


if __name__ == '__main__':
    # todo remove
    arglist = ['--help']

    plac.call(main, arglist=arglist)
