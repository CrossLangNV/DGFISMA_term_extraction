# Should be imported and installed as a package/module
from DGFISMA_RDF.dgfisma_rdf.concepts import build_rdf
from media.eurovoc import get_eurovoc_concepts
from media.wordnet import wordnet_concepts
from similar_terms.glossary import ConceptsVoc
from user_scripts.similar_terms.evaluation import SimTermsConnector

URL = 'http://192.168.105.41:12080'  # Literal 'localhost' won't work


def main():
    """ WordNet will be linked with EuroVoc and exported to an RDF as a linkfile

    Returns:

    """

    concepts_wordnet = ConceptsVoc(wordnet_concepts())
    concepts_eurovoc = ConceptsVoc(get_eurovoc_concepts())

    # Will act as our glossary
    terms_wordnet = concepts_wordnet.get_all_terms()

    # Will act as the vocabulary that wants to be linked
    terms_eurovoc = concepts_eurovoc.get_all_terms()

    sim_terms_connector = SimTermsConnector(URL)
    sim_terms = sim_terms_connector.align(terms_wordnet, terms_eurovoc)

    graph = build_rdf.ConceptGraph()

    l_uri_eurovoc = graph.add_terms(terms_eurovoc)
    l_uri_wordnet = graph.add_terms(terms_wordnet)

    concepts_eurovoc_inverse = {}
    for k, v in concepts_eurovoc.items():
        for v_j in v:
            concepts_eurovoc_inverse.setdefault(v_j, []).append(k)

    sim_concepts = {}
    for concept_wordnet_i, terms_wordnet_i in concepts_wordnet.items():

        l_sim_i = []
        for term_i in terms_wordnet_i:
            terms_eurovoc_j = sim_terms.get(term_i, [])

            for term_eurovoc_j in terms_eurovoc_j:
                for concept_uri_j in concepts_eurovoc_inverse.get(term_eurovoc_j):
                    l_sim_i.append(concept_uri_j)

        if l_sim_i:
            sim_concepts[concept_wordnet_i] = l_sim_i

    graph_link = build_rdf.LinkConceptGraph()
    graph_link.add_similar_terms(sim_concepts)

    s = graph_link.serialize(format="pretty-xml").decode()
    print(s[:1000])  # TODO compare with eurovoc how they save it.
    """
    ...
      </rdf:Description>
      <rdf:Description rdf:about="01791911-a">
        <skos:relatedMatch rdf:resource="http://eurovoc.europa.eu/2389"/>
      </rdf:Description>
      <rdf:Description rdf:about="05004532-n">
        <skos:relatedMatch rdf:resource="http://eurovoc.europa.eu/3352"/>
      </rdf:Description>
      <rdf:Description rdf:about="06642518-n">
        <skos:relatedMatch rdf:resource="http://eurovoc.europa.eu/3825"/>
      </rdf:Description>
      <rdf:Description rdf:about="06625062-n">
        <skos:relatedMatch rdf:resource="http://eurovoc.europa.eu/5340"/>
      </rdf:Description>
      <rdf:Description rdf:about="02046755-v">
        <skos:relatedMatch rdf:resource="http://eurovoc.europa.eu/8552"/>
      </rdf:Description>
      <rdf:Description rdf:about="00926472-v">
        <skos:relatedMatch rdf:resource="http://eurovoc.europa.eu/2614"/>
      </rdf:Description>
      <rdf:Description rdf:about="15244505-n">
        <skos:relatedMatch rdf:resource="http://eurovoc.europa.eu/3988"/>
      </rdf:Description>
    </rdf:RDF>
    """


if __name__ == '__main__':
    main()
