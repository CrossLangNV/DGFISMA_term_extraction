import logging

import numpy as np

from media.eurovoc import get_eurovoc_terms
from similar_terms.methods import SimilarWordsRetriever


def main():
    """
    Evaluate different settings for similarity search of terms.

    Returns:

    """

    # Database
    terms_eurovoc = get_eurovoc_terms()
    # TODO similar terms
    # TODO non-similar terms

    terms_voc = [term_i for l_i in terms_eurovoc.values() for term_i in l_i]

    # Models

    similar_words_retriever_baseline = SimilarWordsRetriever(terms_voc)

    # TODO more models
    models = [similar_words_retriever_baseline]

    # Loop and evaluate different models
    for model in models:

        # pred = model.predict()

        # score = evaluate(pred)

        # TODO save score in better format.
        # print(score)

        l_scores_sim = []

        # Score similar words:
        for uri, l_terms_i in terms_eurovoc.items():

            if len(l_terms_i) >= 2:

                scores_i = []

                # Compare every combination
                for i, l_terms_i_0 in enumerate(l_terms_i):
                    for l_terms_i_j in l_terms_i[i + 1:]:
                        score_j = model.get_sim_between_words(l_terms_i_0, l_terms_i_j)
                        scores_i.append(score_j)

                if 0:
                    print(np.mean(scores_i), l_terms_i)

                l_scores_sim.append(scores_i)

        s = f'Average similarity score for identical terms: {np.mean(scores_i)}. {model}'
        print(s)
        logging.info(s)

    # Conclusion

    return


if __name__ == '__main__':
    main()
