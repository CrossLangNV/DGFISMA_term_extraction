import os

from media.eurovoc import get_eurovoc_concepts, get_eurovoc_related_concepts
from user_scripts.similar_terms.evaluation import eval_script, SimTermsConnector

ROOT = os.path.join(os.path.dirname(__file__), '..')

URL = 'http://192.168.105.41:12080'  # Literal 'localhost' won't work


def main():
    print('EuroVoc dataset')
    concepts = get_eurovoc_concepts()
    related_concepts = get_eurovoc_related_concepts()

    l_terms = [term_i for concept in concepts.values() for term_i in concept]

    sim_terms_connector = SimTermsConnector(URL)
    sim_terms = sim_terms_connector.align_self(l_terms)

    # Remove duplicates and self:
    def remove_duplicates_self(sim_terms):
        for term_i, sim_terms_i in sim_terms.items():
            sim_terms[term_i] = list(set(filter(lambda s: s != term_i, sim_terms_i)))

        return sim_terms

    sim_terms = remove_duplicates_self(sim_terms)

    eval_script(concepts,
                related_concepts,
                sim_terms)


if __name__ == '__main__':
    main()
