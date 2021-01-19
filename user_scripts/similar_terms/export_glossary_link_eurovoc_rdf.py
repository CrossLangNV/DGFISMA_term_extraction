import os
import time
from builtins import staticmethod
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plac
from dgfisma_rdf.concepts import build_rdf
from rdflib.plugins.serializers.turtle import TurtleSerializer
from rdflib.store import Store
from rdflib.term import Literal, Identifier

from media.eurovoc import get_eurovoc_concepts
from similar_terms.glossary import ConceptsVoc
from user_scripts.similar_terms.evaluation import SimTerms

DELIMITER = '⬤'


@plac.annotations(
    filename_terms=("Filename of file with the terms on each row.",),
    filename_rdf=("Filename of where to save rdf to.", "option"),
    b_sim_self=("Flag if similarity with itself also should be computed.", "option")
)
def main(filename_terms: Path,
         filename_rdf: Path = "glossary_link_eurovoc.rdf",
         b_sim_self: bool = True,
         delimiter=DELIMITER,
         ):
    """ Export the glossary (and related terms) to an RDF file as Turtle store.

    Args:
        filename_terms: File where the terms are saved.
            If .csv file with Concepts: [Term, Definition, (Lemma)]
            Else each line we expect a single term as string.
        filename_rdf: Filename where the RDF is saved. Contains the concepts and relationship of related terms with EuroVoc.
        b_sim_self: Boolean if self similarity also has to be predicted
        delimiter: Used delimiter of the CSV

    Returns:
        None
    """

    start_time = time.time()

    # Get terms
    # A concept is defined as a term with a definition
    if os.path.splitext(filename_terms)[-1] in ('.csv', '.CSV'):
        df = csv_glossary_reader(filename_terms, delimiter=delimiter)
        label_term = df.keys()[0]
        l_terms = df[label_term].to_list()
        label_def = df.keys()[1]
        l_def = df[label_def].to_list()
        # # We do nothing with the lemma's at the moment.
        # label_lemma = df.keys()[2]
        # l_lemma = df[label_lemma].to_list()

        l_terms_concept, l_def_concept = zip(
            *[(term_i, def_i) for term_i, def_i in zip(l_terms, l_def) if bool(def_i)])

    else:
        with open(filename_terms, mode='r') as fp:
            l_terms = fp.read().splitlines()

        l_terms_concept, l_def_concept, l_lemma_concept = zip(*[
            (term_i, "", "") for term_i in l_terms
        ])

    print(f'Done loading terms: {(time.time() - start_time):.2f} s')

    with open(filename_rdf, 'w') as file_rdf:
        writer = RDFBufferedWriter(file_rdf)

        if 0:
            graph = build_rdf.LinkConceptGraph()
        else:
            graph = GraphBufferedWriter(writer)

        print(f'\t # terms = {len(l_terms_concept)}')
        l_uri = graph.add_terms(l_terms_concept, l_def=l_def_concept)

        print(f'Done saving terms: {(time.time() - start_time):.2f} s')

        d_uri_terms = {uri: term for term, uri in zip(l_terms_concept, l_uri)}

        # Find similar terms
        sim_terms_connector = SimTerms()

        print(f'Connecting to API: {(time.time() - start_time):.2f} s')

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
            for term_i, l_sim_term_i in similar_terms.items():

                l_uri_sim_i = [uri_j for term_j in l_sim_term_i for uri_j in d_word_uri2.get(term_j)]

                for uri_i in d_word_uri.get(term_i):
                    # Don't add self similarity
                    d_sim_concepts.setdefault(uri_i, []).extend([uri_j for uri_j in l_uri_sim_i if uri_j != uri_i])

            return d_sim_concepts

        # - With itself
        if b_sim_self:
            # Removing duplicates
            sim_terms_self = sim_terms_connector.align_self(list(set(l_terms_concept)))  # l_terms

            print(f'Self similarity prediction: {(time.time() - start_time):.2f} s')
            print(f'\t # self-similar pairs = {sum(map(len, sim_terms_self.values()))}')

            sim_concepts_self = get_similar_concepts(sim_terms_self, d_uri_terms)  # TODO

            print(f'Self similarity URI prep: {(time.time() - start_time):.2f} s')

            print(f'\t # self-similar pairs (uri) = {sum(map(len, sim_concepts_self.values()))}')

            graph.add_similar_terms(sim_concepts_self)

            print(f'Self similarity save: {(time.time() - start_time):.2f} s')

        concepts_eurovoc = ConceptsVoc(get_eurovoc_concepts())
        l_terms_eurovoc = concepts_eurovoc.get_all_terms()
        # - With EuroVoc
        sim_terms_eurovoc = sim_terms_connector.align(l_terms_concept, l_terms_eurovoc)

        print(f'Similarity to EuroVoc prediction: {(time.time() - start_time):.2f} s')
        print(f'\t# EuroVoc pairs = {sum(map(len, sim_terms_eurovoc.values()))}')

        sim_concepts_eurovoc = get_similar_concepts(sim_terms_eurovoc, d_uri_terms, concepts_eurovoc)  # TODO

        print(f'\t # EuroVoc pairs (uri) = {sum(map(len, sim_concepts_eurovoc.values()))}')

        graph.add_similar_terms(sim_concepts_eurovoc)

        print(f'Similarity to EuroVoc save: {(time.time() - start_time):.2f} s')

    return


class RDFBufferedWriter:
    """
    Buffered Writer wrapper on a file, to write the RDF triples to in Turtle format.
    """

    def __init__(self, file: Path):
        self.file = file

        self.serializer = DebuggedTurtleSerializer()

    def write_triple(self, s: Identifier, p: Identifier, o: Identifier):
        """ Very simple Triple writer for fast exporting.

        Args:
            s: Subject
            p: Predicate
            o: Object

        Returns:

        """

        s_str = self.quick_n3(s)
        p_str = self.quick_n3(p)
        o_str = self.quick_n3(o)

        self.file.write(f'{s_str} {p_str} {o_str} .\n')

    @staticmethod
    def quick_n3(node: Identifier):
        """ In order to heavily speed up the export, all assertions are dropped.

        Args:
            node: Graph node.

        Returns:
            String representation for writing to turtle.
        """

        if isinstance(node, Literal):
            return node._literal_n3()
        else:  # URI
            return "<%s>" % node


class DebuggedTurtleSerializer(TurtleSerializer):
    """
    Only used to overwrite a bugged method.
    """

    def __init__(self):
        super(DebuggedTurtleSerializer, self).__init__(Store())

    def getQName(self, *args, **kwar):
        """
        Original method raised and error. We overwrite it to circumvent this error.

        Args:
            *args:
            **kwar:

        Returns:

        """
        return None


def csv_glossary_reader(filename, delimiter=DELIMITER):
    """ Helper for the CSV reader for the glossary.

    Args:
        filename: Filename of CSV
        delimiter:

    Returns:
        pandas dataframe with 2 or 3 columns: (Terms, Definitions, (opt: Lemma))
    """
    df = pd.read_csv(filename, delimiter=delimiter, engine='python',
                     keep_default_na=False)

    return df


class GraphBufferedWriter(build_rdf.LinkConceptGraph):
    """
    The Graph with a buffered writer for building.
    As nothing is saved in memory, it can not be used for querying.
    """

    def __init__(self, writer: RDFBufferedWriter):
        super(GraphBufferedWriter, self).__init__()

        self.writer = writer

    def add(self, triple):
        """ Overwrite the add method to instead of saving in memory, it is written to file.

        Args:
            triple:

        Returns:

        """
        self.writer.write_triple(*triple)

    def serialize(self, *args, **kwargs):
        """ Disabled as we'll be constantly writing

        Args:
            *args:
            **kwargs:

        Returns:

        """
        return


if __name__ == '__main__':
    plac.call(main)
