import json
import os
from typing import List, Dict, Tuple

from SPARQLWrapper import SPARQLWrapper, JSON

ENDPOINT_EUROVOC = 'http://publications.europa.eu/webapi/rdf/sparql'
# Documentation: https://op.europa.eu/en/web/eu-vocabularies/th-top-concept-scheme/-/resource/eurovoc/100141
URI_EUROVOC = 'http://eurovoc.europa.eu/100141'

EN = 'en'


class EurovocSPARQL(SPARQLWrapper):
    def __init__(self):
        super(EurovocSPARQL, self).__init__(ENDPOINT_EUROVOC)
        self.setReturnFormat(JSON)

    def query_list(self, query_string=""""""):
        """

        Args:
            query_string:

        Returns:

        """

        self.setQuery(query_string)

        try:
            ret = self.query()
            # ret is a stream with the results in XML, see <http://www.w3.org/TR/rdf-sparql-XMLres/>
        except Exception as e:
            print(e)

            return

        results = ret.convert()

        l = []

        for result in results["results"]["bindings"]:
            l.append(tuple(result[key]["value"] for key in result))

        return l


def get_eurovoc_concepts(download=False,
                         filename=os.path.join(os.path.dirname(__file__), 'eurovoc_terms.json')):
    """ EuroVoc terms are extracted from open data.

    Args:
        filename: where terms are saved. Generally no need to change this.

    Returns:
        dictionary with key:value the URI and list of English preferred labels
    """

    # Check if exists already, if not: download.
    if download or not os.path.exists(filename):
        query_string = f"""
          PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        SELECT DISTINCT ?c ?label

        FROM <{URI_EUROVOC}>

        WHERE
        {{

        VALUES ?searchLang {{"en"}}
        VALUES ?relation {{skos:prefLabel skos:altLabel}}

        ?c a skos:Concept .    
        ?c ?relation ?label .

        filter ( lang(?label)=?searchLang )
        }}
        """

        l_terms = EurovocSPARQL().query_list(query_string)

        d_terms_temp = _get_terms_dict_from_list(l_terms)

        with open(filename, 'w') as outfile:
            json.dump(d_terms_temp, outfile, indent=4)

    with open(filename, 'r') as json_file:
        d_terms = json.load(json_file)

    return d_terms


def get_eurovoc_related_concepts(download: bool = False,
                                 filename=os.path.join(os.path.dirname(__file__), 'eurovoc_related_concepts.json'),
                                 lang=EN,
                                 b_entailment: bool = True) -> Dict[str, Dict[str, str]]:
    """ relationships between EuroVoc concepts

    Entailment:
    https://www.w3.org/TR/skos-reference/#mapping-cycles-exactMatch
    "<A> skos:related <B> ." entails "<B> skos:related <A> ."
    TODO correctly implement all entailments:

    Args:
        filename:
        download: Boolean
        limit: number of relationships to return
        lang:
        b_entailment: Boolean to decide if extra relationships should be entailed.

    Returns:
        dictionary of term1 URI with dict of {term2 URI: relationship}
    """
    # TODO only getting back "related terms". Less interesting.

    limit = 0

    # Check if exists already, if not: download.
    if download or not os.path.exists(filename):

        query_string = f"""
    
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    
        select ?l1 ?l2 ?c1 ?c2 ?relation
    
        FROM <{URI_EUROVOC}>
    
        where{{
    
        VALUES ?relation {{skos:related skos:broader skos:narrower
                           skos:exactMatch 
                           skos:closeMatch skos:broadMatch
                           skos:mappingRelation skos:narrowMatch skos:relatedMatch
                           }}
    
        VALUES ?searchLang {{ "{lang}" undef }}
    
        ?c1 a skos:Concept .
        ?c1 ?relation ?c2 .
        ?c2 a skos:Concept .
    
        ?c1 skos:prefLabel ?l1 .
        ?c2 skos:prefLabel ?l2 .
    
        filter ( lang(?l1)=?searchLang )
        filter ( lang(?l2)=?searchLang )
        }}
        """

        if limit > 0:
            query_string += f"""
                LIMIT {limit}
                """

        l_query = EurovocSPARQL().query_list(query_string)

        l_related_concepts = [(c1, c2, relation) for (l1, l2, c1, c2, relation) in l_query]

        d_related_concepts = {}

        for (c1, c2, relation) in l_related_concepts:

            d_related_concepts.setdefault(c1, {})[c2] = relation

            if b_entailment:
                if ('related' in relation.lower()) or \
                        ('exactmatch' in relation.lower()):
                    d_related_concepts.setdefault(c2, {})[c1] = relation

        with open(filename, 'w') as outfile:
            json.dump(d_related_concepts, outfile, indent=4)

        del (d_related_concepts)

    with open(filename, 'r') as json_file:
        d_related_concepts = json.load(json_file)

    return d_related_concepts


def query_different_relationships_concepts() -> str:
    """ Returns the (SKOS) relationships as found in EuroVoc
    https://op.europa.eu/en/web/eu-vocabularies/alignments

    Returns:
        SPARQL query.
    """

    query_string = f"""

       PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

       select distinct ?relation

       FROM <{URI_EUROVOC}>

       where{{

       ?c1 a skos:Concept .
       ?c1 ?relation ?c2 .
       ?c2 a skos:Concept .
       }}
       """
    return query_string


def _get_terms_dict_from_list(l_terms: List[Tuple[str]]) -> Dict[str, List[str]]:
    """ Transform list of (uri, label) pairs to equivalent dict.

    Args:
        l_terms:

    Returns:
        Dictionary with uri: list of labels with same uri.
    """
    # len(l_terms) not matching len(d_terms_temp)
    d_terms = {}
    for (uri, label) in l_terms:
        d_terms.setdefault(uri, []).append(label)

    return d_terms
