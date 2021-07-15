import os
import tempfile
import unittest

from rdflib import Graph, Literal

from user_scripts.similar_terms import export_glossary_link_eurovoc_rdf
from user_scripts.similar_terms.export_glossary_link_eurovoc_rdf import RDFBufferedWriter, _clean_string

TERMS = ['word', 'second', 'the word', 'seconds', 'word', 'another one', 'economic', 'money', 'moneys', 'wordy',
         'public relationships', 'relationships', 'economical transactions', 'Financial transactions']
DEFS = ['', 'SeCoNd', 'the_word', '2nd', 'word', '', 'eco', 'gold', 'Plural?', '+y', 'abc', 'abc', 'abc', 'abc']
LEMMA = ['word', 'second', 'word', 'second', 'word', 'another one', 'economic', 'money', 'money', 'word',
         'public relationship', 'relationship', 'economical transaction', 'financial transaction']
STATE = ['Unvalidated'] * 14


class TestCleanString(unittest.TestCase):
    def test_clean_string(self):
        s_in = 'A normal string'
        s_out = s_in
        s_pred = _clean_string(s_in)

        self.assertEqual(s_pred, s_out, 'Not the expected output.')

    def test_clean_string_newlines(self):
        s_in = 'A normal string\n over multiple lines'
        s_out = 'A normal string over multiple lines'
        s_pred = _clean_string(s_in)

        self.assertEqual(s_pred, s_out, 'Not the expected output.')

    def test_clean_string_newlines_windows(self):
        s_in = 'A normal string\r\n over multiple lines'
        s_out = 'A normal string over multiple lines'
        s_pred = _clean_string(s_in)

        self.assertEqual(s_pred, s_out, 'Not the expected output.')

    def test_clean_string_newlines_mac(self):
        s_in = 'A normal string\r over multiple lines'
        s_out = 'A normal string over multiple lines'
        s_pred = _clean_string(s_in)

        self.assertEqual(s_pred, s_out, 'Not the expected output.')

    def test_clean_string_double_spaces(self):
        s_in = ' All double   spaces    should be gone   !'
        s_out = 'All double spaces should be gone !'
        s_pred = _clean_string(s_in)

        self.assertEqual(s_pred, s_out, 'Not the expected output.')


class TestPipeline(unittest.TestCase):
    def test_full_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            filename_terms = os.path.join(tmp_dir, 'glossary.txt')

            filename_rdf = os.path.join(tmp_dir, 'glossary_link_eurovoc.rdf')

            with open(filename_terms, mode='w') as fp:
                fp.writelines('\n'.join(TERMS) + '\n')

            export_glossary_link_eurovoc_rdf.main(filename_terms, filename_rdf=filename_rdf)

            g = Graph()
            g.parse(filename_rdf, format='turtle')

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

            for term in TERMS:
                self.assertIn(term, objects_str, 'Did not find term in RDF graph')

            with self.subTest('Related terms'):
                triple_filtered = [s for s, p, o in triples if
                                   str(p) == 'http://www.w3.org/2004/02/skos/core#relatedMatch']

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
                self.assertEqual(r_labels, set(TERMS), 'Should return same terms')

            # Get related terms self

            q_self_related = """
            prefix skos: <http://www.w3.org/2004/02/skos/core#>
    
            SELECT ?term0 ?term1 ?subject0 ?subject1
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
                self.assertGreater(len(r_related_all), len(r_sim_term),
                                   f'Should find relations with EuroVoc: {r_related_all}')


class TestCSV(unittest.TestCase):
    def test_full_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            filename_glossary = os.path.join(tmp_dir, 'glossary.csv')

            with open(filename_glossary, mode='w') as fp:
                delimiter = '⬤'
                s_lines = ''
                for term_i, def_i, lemma, state in zip(TERMS, DEFS, LEMMA, STATE):
                    s_lines += delimiter.join([term_i, def_i, lemma, state]) + '\n'

                fp.writelines(s_lines)
            filename_rdf = os.path.join(tmp_dir, 'glossary_link_eurovoc.turtle')

            export_glossary_link_eurovoc_rdf.main(filename_glossary, filename_rdf=filename_rdf)

            g = Graph()
            g.parse(filename_rdf, format='turtle')

        q = """
        SELECT ?subject ?predicate ?object
        WHERE {
          ?subject ?predicate ?object
        }
        # LIMIT 25  # commented out ;)
        """

        triples = list(g.query(q))

        with self.subTest('non-empty'):
            self.assertTrue(len(triples), 'No triples found in RDF Graph.')

        with self.subTest('Find terms'):
            objects_str = [str(o) for s, p, o in triples if 'prefLabel' in p]

            term_with_def = [term_i for term_i, def_i in zip(TERMS, DEFS) if def_i]

            self.assertEqual(sorted(objects_str), sorted(term_with_def),
                             'All terms with a definition should be retrieved')

        with self.subTest('Related terms'):
            triple_filtered = [(s, o) for s, p, o in triples if
                               str(p) == 'http://www.w3.org/2004/02/skos/core#relatedMatch']

            self.assertTrue(len(triple_filtered), 'Did not find related terms in RDF graph')

        # Get related terms self

        with self.subTest('Self similar terms'):
            q_self_related = """
            prefix skos: <http://www.w3.org/2004/02/skos/core#>

            SELECT ?term0 ?term1
            WHERE {
                ?subject0 skos:prefLabel ?term0 ;
                    skos:relatedMatch ?subject0 .
            }
            """

            r_sim_term = list(g.query(q_self_related))

            self.assertFalse(len(r_sim_term), f"Shouldn't retrieve self similarity")

        with self.subTest('Find related terms'):
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

            self.assertTrue(len(r_sim_term), f'Should be non-empty: {r_sim_term}')

        with self.subTest('Find all pairs'):
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

            self.assertGreater(len(r_related_all), len(r_sim_term),
                               f'Should find relations with EuroVoc: {r_related_all}')


class TestRDFBufferedWriter(unittest.TestCase):
    def test_init(self):

        s = Literal('s')
        p = Literal('p')
        o = Literal('o')

        g_comparison = Graph()
        g_comparison.add((s, p, o))

        with tempfile.TemporaryDirectory() as tmp_dir:
            filename_turtle = os.path.join(tmp_dir, 'example.turtle')

            with open(filename_turtle, 'w') as file_rdf:
                writer = RDFBufferedWriter(file_rdf)

                writer.write_triple(s, p, o)

            filename_comparison = os.path.join(tmp_dir, 'example_comparison.rdf')

            # g_comparison.serialize(filename_comparison, format='turtle')

            # with open(filename_comparison) as file_comp:
            #     a = file_comp.read()
            # print(a)

            g = Graph()
            g.parse(filename_turtle, format='turtle')

        with self.subTest('Amount of triples'):
            self.assertEqual(len(g), 1, 'Only expect one triple')

        with self.subTest('Triple value'):
            self.assertEqual(list(g)[0], (s, p, o))

        with self.subTest('Identical graph'):
            self.assertEqual(list(g), list(g_comparison))

        return

    def test_identical_graphs(self):

        g_baseline = Graph()
        g_baseline.parse(os.path.join(os.path.dirname(__file__), 'reporting_obligations_mockup.rdf'))

        with tempfile.TemporaryDirectory() as tmp_dir:

            filename_turtle = os.path.join(tmp_dir, 'example.turtle')

            with open(filename_turtle, 'w') as file_rdf:
                writer = RDFBufferedWriter(file_rdf)

                for (s, p, o) in g_baseline:
                    writer.write_triple(s, p, o)

            g = Graph()
            g.parse(filename_turtle, format='turtle')

        with self.subTest('Amount of triples'):
            self.assertEqual(len(g), len(g_baseline), 'Should have same amount of triples.')

        with self.subTest('Retrieve all triples'):

            for triple in g_baseline:
                self.assertIn(triple, g, 'did not find baseline triple in new graph')

            for triple in g:
                self.assertIn(triple, g_baseline, 'did not find new triple in baseline graph')

        return
