import logging
import os

import fasttext
import numpy as np

from examples.evaluate_simularity import get_pairs_abbreviation
from similar_terms.methods import SimilarWordsRetriever, get_similarity_fast
from similar_terms.preprocessing import preprocessing_word


def init_logger(model_directory):
    """Besides printing intermediate results to the command line, it is also saved in a log file.
    call logger.info("<String to write to log.txt>") instead of print.
    Args:
        model_directory: directory where the log.txt is saved.
    Returns:
        The logger object
    """
    log = logging.getLogger()
    log.handlers.clear()
    formatter = logging.Formatter('%(message)s')
    fh = logging.FileHandler(os.path.join(model_directory, "log.txt"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    log.setLevel(logging.INFO)
    log.addHandler(ch)

    return log


def eval_models():
    """ Small script that evaluates different models (with and without preprocessing terms)

    Returns:None

    """

    log = init_logger('.')
    # logging.basicConfig(filename='./log.txt', format='%(name)s - %(levelname)s - %(message)s')

    sim_words_retriever = SimilarWordsRetriever([])

    ROOT = os.path.join(os.path.dirname(__file__), '..')

    # Evaluation data
    l = get_pairs_abbreviation()

    # preprocessing
    preprocesser = lambda s: preprocessing_word(s, False)

    foo = {}
    for model_path in (
            'media/dgf-model.tok.bin',  # sentence vector embedding n = 100, in [0; 1]
            'media/models_francois/vecs-d3+2-base-skip-b2mn9m5-256m15w6i15.bin',
            # embedding n = 256, in  in [-1; 1]
            'media/models_francois/vecs-d3+2-cbow-b8mn9m5-256m15w6i20.bin',
            # embedding n = 256, in [-1; 1], normalised
            'media/models_francois/vecs-d3-base-cbow-b2mn9m4-256m15w6i15.bin',
            # embedding n = 256, in [-1; 1], normalised
            'media/models_francois/vecs-d3-base-skip-b2mn9m4-256m15w6i15.bin',
            # embedding n = 256, in [-1; 1], normalised
    ):
        sim_words_retriever._ftModel = fasttext.load_model(os.path.join(ROOT, model_path))

        for name, pre_f in {'default preprocesser': preprocesser,
                            'nothing': lambda s: s}.items():

            score = []

            for s1, s2 in l:
                s1, s2 = map(pre_f, (s1, s2))

                e1 = sim_words_retriever.get_embedding(s1)
                e2 = sim_words_retriever.get_embedding(s2)

                score_i = get_similarity_fast(e1, e2)

                score.append(np.nan_to_num(score_i, nan=0.))

            foo[name + ' ' + model_path] = np.mean(score)

    foo_sorted = {k: v for k, v in sorted(foo.items(), key=lambda item: item[1], reverse=True)}
    log.info('sorted models:')
    log.info(foo_sorted)


if __name__ == '__main__':
    eval_models()
