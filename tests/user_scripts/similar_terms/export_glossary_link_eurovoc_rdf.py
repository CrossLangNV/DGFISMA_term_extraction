import os
import tempfile
import unittest

from rdflib import Graph

from user_scripts.similar_terms import export_glossary_link_eurovoc_rdf


class TestPipeline(unittest.TestCase):
    def test_full_pipeline(self):
        l_terms = ['word', 'second', 'the word', 'secondary', 'word', 'another one', 'economic', 'money']

        with tempfile.TemporaryDirectory() as tmp_dir:
            filename_terms = os.path.join(tmp_dir, 'glossary.txt')

            filename_rdf = os.path.join(tmp_dir, 'glossary_link_eurovoc.rdf')

            with open(filename_terms, mode='w') as fp:
                fp.writelines('\n'.join(l_terms) + '\n')

            export_glossary_link_eurovoc_rdf.main(filename_terms, filename_rdf=filename_rdf)

            g = Graph()
            g.parse(filename_rdf)

        q = """
        SELECT ?subject ?predicate ?object
        WHERE {
          ?subject ?predicate ?object
        }
        # LIMIT 25
        """

        triples = list(g.query(q))

        with self.subTest('non-empty'):
            self.assertTrue(len(triples), 'No triples found in RDF Graph.')

        with self.subTest('Find terms'):
            objects_str = [str(o) for s, p, o in triples]

            for term in l_terms:
                self.assertIn(term, objects_str, 'Did not find term in RDF graph')

        with self.subTest('Related terms'):
            triple_filtered = [s for s, p, o in triples if str(p) == 'http://www.w3.org/2004/02/skos/core#relatedMatch']

            self.assertTrue(len(triple_filtered), 'Did not find related terms in RDF graph')

        # Get all terms
        q_terms = """
        prefix skos: <http://www.w3.org/2004/02/skos/core#>

        SELECT ?term
        WHERE {
          ?subject skos:prefLabel ?term .
        }
        """

        r_labels = set(map(str, [el[0] for el in g.query(q_terms)]))

        with self.subTest('Find terms'):
            self.assertEqual(r_labels, set(l_terms), 'Should return same terms')


        # Get related terms self

        q_self_related = """
        prefix skos: <http://www.w3.org/2004/02/skos/core#>

        SELECT ?term0 ?term1
        WHERE {
            ?subject0 skos:prefLabel ?term0 ;
                skos:relatedMatch ?subject1 .
            FILTER ( ?subject0 != ?subject1 )
            ?subject1 skos:prefLabel ?term1 .

        }
        """

        r_sim_term = list(g.query(q_self_related))

        with self.subTest('Find related terms'):
            self.assertTrue(len(r_sim_term), f'Should be non-empty: {r_sim_term}')


        q_related_all = """
        prefix skos: <http://www.w3.org/2004/02/skos/core#>

        SELECT ?term0 ?subject1
        WHERE {
            ?subject0 skos:prefLabel ?term0 ;
                skos:relatedMatch ?subject1 .
            FILTER ( ?subject0 != ?subject1 )
        }
        """

        r_related_all = list(g.query(q_related_all))

        with self.subTest('Find all pairs'):
            self.assertGreater(len(r_related_all), len(r_sim_term), f'Should find relations with EuroVoc: {r_related_all}')
