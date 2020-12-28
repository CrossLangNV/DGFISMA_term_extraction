import os
import sys
from pathlib import Path
from typing import Dict, List

import plac

from media.eurovoc import get_eurovoc_concepts
from similar_terms.glossary import ConceptsVoc
from user_scripts.similar_terms.evaluation import SimTermsConnector

# import DGFISMA_RDF
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../DGFisma_rdf')))
from concepts import build_rdf

URL = 'http://192.168.105.41:12080'  # Literal 'localhost' won't work


@plac.annotations(
    filename_terms=("Filename of file with the terms on each row.",),
    filename_rdf=("Filename of where to save rdf to.", "option"),
    b_sim_self=("Flag if similarity with itself also should be computed.", "option")
)
def main(filename_terms: Path,
         filename_rdf: Path = "glossary_link_eurovoc.rdf",
         b_sim_self: bool = True,
         ):
    """

    Args:
        filename_terms: Text file where the terms are saved. On each line we expect a single term as string.
        filename_rdf: Filename where the RDF is saved. Contains the concepts and relationship of related terms with EuroVoc.

    Returns:

    """

    # Get terms

    with open(filename_terms, mode='r') as fp:
        l_terms = fp.read().splitlines()

    graph = build_rdf.LinkConceptGraph()

    l_uri = graph.add_terms(l_terms)

    d_uri_terms = {uri: term for term, uri in zip(l_terms, l_uri)}

    # Find similar terms
    sim_terms_connector = SimTermsConnector(URL)

    def get_similar_concepts(similar_terms: Dict[str, List[str]], concepts: Dict[str, str], concepts2=None):
        """

        Args:
            similar_terms:
            concepts:
            concepts2: Optional, if linked with another glossary

        Returns:

        """

        if concepts2 is None:
            concepts2 = concepts

        def get_d_word_uri(concepts):
            # Quick lookup
            d_word_uri = {}
            for uri, term in concepts.items():

                if isinstance(term, list):
                    for term_i in term:
                        d_word_uri.setdefault(term_i, []).append(uri)
                else:
                    d_word_uri.setdefault(term, []).append(uri)
            return d_word_uri

        d_word_uri = get_d_word_uri(concepts)
        d_word_uri2 = get_d_word_uri(concepts2)

        d_sim_concepts = {}
        # TODO remove duplicates.
        for term_i, l_sim_term_i in similar_terms.items():

            l_uri_sim_i = [uri_j for term_j in l_sim_term_i for uri_j in d_word_uri2.get(term_j)]

            for uri_i in d_word_uri.get(term_i):
                d_sim_concepts.setdefault(uri_i, []).extend(l_uri_sim_i)

        return d_sim_concepts

    # - With itself
    if b_sim_self:
        sim_terms_self = sim_terms_connector.align(l_terms, l_terms)

        sim_concepts_self = get_similar_concepts(sim_terms_self, d_uri_terms)  # TODO

        graph.add_similar_terms(sim_concepts_self)

    concepts_eurovoc = ConceptsVoc(get_eurovoc_concepts())
    l_terms_eurovoc = concepts_eurovoc.get_all_terms()
    # - With EuroVoc
    sim_terms_eurovoc = sim_terms_connector.align(l_terms, l_terms_eurovoc)

    sim_concepts_eurovoc = get_similar_concepts(sim_terms_eurovoc, d_uri_terms, concepts_eurovoc)  # TODO
    graph.add_similar_terms(sim_concepts_eurovoc)

    # Save to RDF
    # Export to file
    graph.serialize(destination=filename_rdf, format="pretty-xml")

    return


if __name__ == '__main__':
    plac.call(main)
