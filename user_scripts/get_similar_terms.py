from pathlib import Path

import plac

from similar_terms.methods import SimilarWordsRetriever


# @plac.pos('path_term', "Path to file with term")
# @plac.pos('path_terms_voc', "Path to file with vocabulary")
# @plac.pos('path_out', "Path to write similar terms to")

# @plac.annotations(
#     path_term=plac.Annotation(type=Path)
#     # ('model', "Model name", Path)
# )
def main(
        path_term,
        path_terms_voc,  #: Path,
        path_out  #: Path
):
    """

    Args:
        path_term: asdf
        path_terms_voc: asdf
        path_out:as fd

    Returns: asdf
    """
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
