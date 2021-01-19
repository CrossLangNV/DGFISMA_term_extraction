import json
import os
import time
from random import sample, seed
from typing import List

import numpy as np

from media.wordnet import wordnet_concepts, wordnet_related_concepts
from user_scripts.similar_terms.evaluation import eval_script, SimTermsConnector

URL = 'http://192.168.105.41:12080'  # Literally 'localhost' won't work

FILENAME_TMP = os.path.join(os.path.dirname(__file__), '../../media/tmp/sim_terms_tmp.json')


def eval_speed(l_terms: List[str],
               connector: SimTermsConnector,
               N_SUB=1,
               n_iter=1,
               seed_val=123):
    seed(seed_val)  # set seed for reproducibility

    t_lst = []
    for _ in range(n_iter):
        l_terms_sub = sample(l_terms, N_SUB)

        start_time = time.time()
        _ = connector.align_self(l_terms_sub)
        end_time = time.time()

        t_lst.append(end_time - start_time)

    print(f'Similar term retrieval took: {np.mean(t_lst):.5f} +- {np.std(t_lst, ddof=1):.5f} seconds')


def main():
    # Get data
    print('WordNet data')
    concepts = wordnet_concepts()

    related_concepts = wordnet_related_concepts()

    l_terms = [term_i for concept in concepts.values() for term_i in concept]

    N_SUB = 1000
    l_terms_sub = [l_terms[i] for i in range(N_SUB)]  # for testing purposes

    # Get similar terms
    # - Make a Connector
    sim_terms_connector = SimTermsConnector(URL)

    b = 0
    if b:
        eval_speed(l_terms, sim_terms_connector,
                   N_SUB=1000,
                   n_iter=100)

    b = 0
    if b:
        # Split up:
        start_time = time.time()
        sim_terms = {}
        for i_start in range(0, len(l_terms), N_SUB):
            print(f'\t{i_start}/{len(l_terms)}')
            l_terms_sub = l_terms[i_start:i_start + N_SUB]

            sim_terms_i = sim_terms_connector.align(l_terms_sub, l_terms)
            sim_terms.update(sim_terms_i)

        end_time = time.time()

        print(f'Similar term retrieval took: {(end_time - start_time):.5f} seconds')

    b = False
    if b:
        with open(FILENAME_TMP, 'w') as outfile:
            json.dump(sim_terms, outfile, indent=4)

    with open(FILENAME_TMP, 'r') as json_file:
        sim_terms = json.load(json_file)

    # Remove duplicates and self:
    def remove_duplicates_self(sim_terms):
        for term_i, sim_terms_i in sim_terms.items():
            sim_terms[term_i] = list(set(filter(lambda s: s != term_i, sim_terms_i)))

        return sim_terms

    sim_terms = remove_duplicates_self(sim_terms)

    eval_script(concepts,
                related_concepts,
                sim_terms)

    return


if __name__ == '__main__':
    main()
