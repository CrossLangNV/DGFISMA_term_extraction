import json
import os
from typing import List, Dict, Tuple

from SPARQLWrapper import SPARQLWrapper, JSON

ENDPOINT_EUROVOC = 'http://publications.europa.eu/webapi/rdf/sparql'
# https://op.europa.eu/en/web/eu-vocabularies/th-top-concept-scheme/-/resource/eurovoc/100141
URI_EUROVOC = 'http://eurovoc.europa.eu/100141'

EN = 'en'


# def get_query_concept_by_example(term: str,  # 'term'
#                                  lang: str = EN
#                                  ):
#     """ Query returns concept URI and label
#
#     Args:
#         term:
#         lang:
#
#     Returns:
#
#     """
#
#     assert len(term)
#     assert len(lang)
#
#     query_string = f"""
#         PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
#         select ?c ?label
#
#         FROM <{URI_EUROVOC}>
#
#         where
#         {{
#
#         VALUES ?searchTerm {{"{term}"}}
#         VALUES ?searchLang {{"{lang}"}}
#
#         VALUES ?relation {{skos:prefLabel skos:altLabel}}
#         ?c a skos:Concept .
#         ?c ?relation ?label .
#
#         filter ( contains(?label,?searchTerm) && lang(?label)=?searchLang )
#         }}
#         """
#
#     return query_string
#
#
# def get_query_related_terms(lang: str = EN,
#                             limit: int= 0):
#
#     query_string = f"""
#
#     PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
#
#     select ?l1 ?l2 ?c ?sibling
#
#     FROM <{URI_EUROVOC}>
#
#     where{{
#
#     # VALUES ?c {{ "{"EuroVoc concept URI"}" }}
#     VALUES ?relation {{ skos:related }} # skos:broader
#     VALUES ?searchLang {{ "{lang}" undef }}
#
#     ?c a skos:Concept .
#     ?c ?relation ?sibling .
#
#     ?c skos:prefLabel ?l1 .
#     ?sibling skos:prefLabel ?l2 .
#
#     filter ( lang(?l1)=?searchLang )
#     filter ( lang(?l2)=?searchLang )
#     }}
#     """
#
#     if limit > 0:
#         query_string += f"""
#             LIMIT {limit}
#             """
#
#     return query_string


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


def get_eurovoc_terms(download=False,
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

        d_terms_temp = get_terms_dict_from_list(l_terms)

        with open(filename, 'w') as outfile:
            json.dump(d_terms_temp, outfile, indent=4)

    with open(filename, 'r') as json_file:
        d_terms = json.load(json_file)

    return d_terms


def get_terms_dict_from_list(l_terms: List[Tuple[str]]) -> Dict[str, List[str]]:
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


if __name__ == '__main__':
    get_data()
