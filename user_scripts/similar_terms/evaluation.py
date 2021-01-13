import urllib.parse
from typing import List

import numpy as np
import requests


def eval_script(concepts, related_concepts, sim_terms_pred):
    # Statistics
    n_same_concept = [len(same_concepts_i) for same_concepts_i in concepts.values()]
    n_same_concept_avg = np.mean(n_same_concept)
    print(f'Average number of terms for same concept: {n_same_concept_avg}.')

    # - number of similar terms
    n_sim_concepts = [len(sim_concepts_i) for sim_concepts_i in sim_terms_pred.values()]
    n_sim_concepts_avg = np.mean(n_sim_concepts)
    print(f'Average number of predicted similar terms: {n_sim_concepts_avg}.')

    # - number of identical terms
    # - Accuracies?

    n_tp = 0  # number of similar term retrieved
    n_fp = 0  # number of retrieved terms that are not similar
    n_fn = 0  # number of similar terms that are not retrieved

    print('Evaluation on same concepts')
    for concept_i, l_terms_i in concepts.items():
        for term_i in l_terms_i:
            other_terms_i = [term_j for term_j in l_terms_i if term_j != term_i]

            s_other_terms = set(other_terms_i)

            sim_terms_i = sim_terms_pred.get(term_i)
            s_sim_terms = set(sim_terms_i)

            n_tp += len(s_other_terms.intersection(s_sim_terms))
            n_fp += len(s_sim_terms - s_other_terms)
            n_fn += len(s_other_terms - s_sim_terms)

    print(f'# TP: {n_tp}')
    print(f'# FP: {n_fp}')
    print(f'# FN: {n_fn}')

    # Similar terms

    print('Evaluation on same hypo- & hypernyms')
    n_tp = 0  # number of similar term retrieved
    n_fp = 0  # number of retrieved terms that are not similar
    n_fn = 0  # number of similar terms that are not retrieved
    for concept_id, related_concepts_i in related_concepts.items():

        l_terms_i = concepts.get(concept_id)

        for term_i in l_terms_i:
            other_terms_i = [term_j for id_j in related_concepts_i.keys() for term_j in concepts.get(id_j)]

            s_other_terms = set(other_terms_i)

            sim_terms_i = sim_terms_pred.get(term_i)
            s_sim_terms = set(sim_terms_i)

            n_tp += len(s_other_terms.intersection(s_sim_terms))
            n_fp += len(s_sim_terms - s_other_terms)
            n_fn += len(s_other_terms - s_sim_terms)

    print(f'# TP: {n_tp}')
    print(f'# FP: {n_fp}')
    print(f'# FN: {n_fn}')


class SimTermsConnector:
    """
    Connects to the Similar Terms FastAPI
    """

    def __init__(self, url):
        r_homepage = requests.get(url)  # Homepage
        r_docs = requests.get(urllib.parse.urljoin(url, 'docs'))  # Swagger

        assert r_homepage.status_code < 300 or r_docs.status_code < 300  # Check if API is accessible

        self.url = url

    def align(self, voc1: List[str], voc2: List[str]):
        """ Aligns 2 vocs based on similarity of words.

        Args:
            voc1:
            voc2:

        Returns:

        """

        r_docs = requests.post(urllib.parse.urljoin(self.url, 'similar_terms/align'),
                               json={'voc1': list(set(voc1)),
                                     'voc2': list(set(voc2))})  # Swagger

        assert r_docs.status_code < 300

        sim_terms = r_docs.json()

        return sim_terms

    def align_self(self, voc: List[str]):
        """ Align vocabulary with itself based on similar words.

        Args:
            voc:

        Returns:

        """

        r_docs = requests.post(urllib.parse.urljoin(self.url, 'similar_terms/self'),
                               json={'voc': list(set(voc))})  # Swagger

        assert r_docs.status_code < 300

        sim_terms = r_docs.json()

        return sim_terms
