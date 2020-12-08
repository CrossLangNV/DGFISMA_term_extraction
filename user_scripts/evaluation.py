import logging
import os
import random
from typing import List, Tuple

import fasttext
import numpy as np
from sklearn.metrics import confusion_matrix

from media.eurovoc import get_eurovoc_concepts, get_eurovoc_related_concepts
from similar_terms.methods import SimilarWordsRetriever
from similar_terms.preprocessing import preprocessing_word

ROOT = os.path.join(os.path.dirname(__file__), '..')


def build_same_concept_data(seed=123,
                            n_samples=10000
                            ) -> (List[Tuple[str]], List[int]):
    concepts_eurovoc = get_eurovoc_concepts()

    # [((label 1, label 2), 1), ...]. 1 because all matches
    l_data = []

    random.seed(seed)

    sampler1 = _random_sample_generator(concepts_eurovoc.items(), n_samples)  # random.sample(, )

    for (concept_i, l_labels_i) in sampler1:

        n_labels_i = len(l_labels_i)

        if n_labels_i <= 1:
            continue  # no multiple labels

        j = random.choice(range(n_labels_i))

        label_i = l_labels_i[j]
        label_j = random.choice(l_labels_i[:j] + l_labels_i[j + 1:])

        l_data.append(((label_i, label_j), 1))

    evaluation_data, evaluation_labels = zip(*l_data)

    return evaluation_data, evaluation_labels


def build_evaluation_data(seed=123,
                          n_samples=100000  # 100k
                          ):
    """ List of pairs of labels and if they are related or not.
    This list is seeded for reproducibility

    Args:
        seed:
        n_samples:

    Returns:

    """

    concepts_eurovoc = get_eurovoc_concepts()

    related_concepts = get_eurovoc_related_concepts()

    # [((label 1, label 2), 1 if match else 0), ...]
    l_data = []

    random.seed(seed)

    sampler1 = _random_sample_generator(concepts_eurovoc.items(), n_samples)  # random.sample(, )
    sampler2 = _random_sample_generator(concepts_eurovoc.items(), n_samples)

    for (concept_i, l_labels_i), (concept_j, l_labels_j) in zip(sampler1, sampler2):
        if concept_i == concept_j:
            continue  # ignore if the same concept.

        label_i = random.choice(l_labels_i)
        label_j = random.choice(l_labels_j)

        related_concepts_i = related_concepts.get(concept_i, {})

        match = int(concept_j in related_concepts_i)

        l_data.append(((label_i, label_j), match))

    evaluation_data, evaluation_labels = zip(*l_data)

    return evaluation_data, evaluation_labels


def build_eval_data_balanced(seed=123,
                             n_samples=100000  # 100k
                             ):
    """ List of pairs of labels and if they are related or not.
    More balanced dataset.

    Args:
        seed:     This list is seeded for reproducibility.
        n_samples:

    Returns:

    """

    evaluation_data_1, evaluation_labels_1 = build_same_concept_data(n_samples=n_samples // 2, seed=seed)

    evaluation_data_unbalanced, evaluation_labels_unbalanced = build_evaluation_data(n_samples=n_samples // 2,
                                                                                     seed=seed)

    evaluation_data = evaluation_data_1 + evaluation_data_unbalanced
    evaluation_labels = evaluation_labels_1 + evaluation_labels_unbalanced

    return evaluation_data, evaluation_labels


def decide_threshold(model,
                     voc: List[str] = None,
                     k_similar: int = None) -> float:
    """ Threshold such that on average(/median) k related terms
    TODO Question: How much related terms in EuroVoc on average?

    Find k nearest terms. What is the cutoff similarity? Take average cutoff.

    Args:
        model:
        voc:
        k_similar: TODO might be deprecated as k is decided based on average number of related terms in EuroVoc

    Returns:
        Cutoff similarity score
    """

    # TODO remove
    concepts = get_eurovoc_concepts()
    related = get_eurovoc_related_concepts()

    k_avg_identical = np.mean([len(l_terms) for l_terms in concepts.values()])

    k_avg_related_concepts = np.mean([len(related.get(concept, {})) for concept in concepts])

    l_k_matches = []
    for concept_i in concepts:
        l_concept_related_i = related.get(concept_i, {})
        n_terms_related_i = [len(concepts[concept_j]) for concept_j in l_concept_related_i]

        # number of related terms is number of other terms with same concept and 'similar' concepts
        k_matches = len(concepts[concept_i]) - 1 + sum(n_terms_related_i)

        l_k_matches.append(k_matches)

    k_avg_related_terms = np.mean(l_k_matches)

    print(f'average number of related terms: {k_avg_related_terms}')

    # Find related terms
    voc_terms = [term_i for concept_i in concepts.values() for term_i in concept_i]

    model.set_vocabulary(voc_terms)

    k_similar = round(k_avg_related_terms)

    l_thresh = []
    for term_i in voc_terms:
        a = model.get_similar_k(term_i, k=k_similar, include_self=False)
        thresh_min_i = min(a.get('score'))
        l_thresh.append(thresh_min_i)

    thresh_avg_k = np.mean(l_thresh)

    # TODO validate that when threshold on you get on average k terms back.

    return thresh_avg_k


def main():
    """
    Evaluate different settings for similarity search of terms.

    Returns:

    """

    # Database

    if 1:
        print('Unbalanced evaluation data')
        evaluation_data, evaluation_labels = build_evaluation_data()
    else:
        print('Same concept data')
        evaluation_data, evaluation_labels = build_same_concept_data()

    evaluation_data_balanced, evaluation_labels_balanced = build_eval_data_balanced(n_samples=10000)

    # Models
    models = {
        'baseline': SimilarWordsRetriever(),
        'Casing': SimilarWordsRetriever([], preprocessor=lambda s: preprocessing_word(s, b_lower=False)),
        'Casing + spaces': SimilarWordsRetriever([], preprocessor=lambda s: preprocessing_word(s, b_lower=False,
                                                                                               b_join=False))
    }
    # add all models from Francois
    if 1:
        folder_models = os.path.join(ROOT, "media/models_francois")
        for file in os.listdir(folder_models):

            if file.endswith(".bin"):
                filename = os.path.join(folder_models, file)
            else:
                continue

            for b_join in [True, False]:
                similar_words_retriever_i = SimilarWordsRetriever([], preprocessor=lambda s,
                                                                                          b_join=b_join: preprocessing_word(
                    s,
                    b_join=b_join,
                    b_lower=False))
                similar_words_retriever_i._ftModel = fasttext.load_model(filename)

                s_join = ' "_"-joining' if b_join else ''
                models[file + s_join] = similar_words_retriever_i

    # Loop and evaluate different models
    for model_name, model in models.items():
        print(f'Model: {model_name}')

        if 1:
            thresh_avg_k = decide_threshold(model, voc=get_eurovoc_related_concepts())
            print(f'\tThresh avg k: {thresh_avg_k}')
        else:
            # Just simple rounding
            thresh_avg_k = 0.5

        eval_model(model,
                   evaluation_data_balanced,
                   evaluation_labels_balanced,
                   thresh=thresh_avg_k)

        print()

    return


def eval_model(model, x, labels, thresh=.5):
    l_scores_sim = []
    for (label1, label2) in x:
        score_j = model.get_sim_between_words(label1, label2)
        l_scores_sim.append(score_j)

    pred_bool = np.greater_equal(l_scores_sim, thresh)
    conf = confusion_matrix(labels, pred_bool)
    logging.info(f'Confusion matrix:\n{conf}')
    print(f'Confusion matrix:\n{conf}')

    # Examples:
    for i, (label_i, pred_i) in enumerate(zip(labels, pred_bool)):
        if label_i and pred_i:
            print(f'{x[i]}\n{(label_i, pred_i)}')
            break


def _random_sample_generator(l: list,
                             n_max: int):
    l_list = list(l)

    for i in range(n_max):
        yield random.choice(l_list)


if __name__ == '__main__':
    main()
