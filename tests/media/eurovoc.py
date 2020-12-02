import os
import tempfile
import unittest

from media.eurovoc import get_eurovoc_terms, EurovocSPARQL, URI_EUROVOC, get_terms_dict_from_list


class TestEurovocSPARQL(unittest.TestCase):

    def test_find_triples(self):

        LIMIT = 11
        q = f"""
        SELECT ?s ?p ?o

        WHERE {{?s ?p ?o}}
        
        LIMIT {LIMIT}
        """

        l = EurovocSPARQL().query_list(q)

        with self.subTest('limit'):
            self.assertEqual(LIMIT, len(l))

        with self.subTest('Type'):
            self.assertIsInstance(l, list, 'Should return a list')

        with self.subTest('content'):
            for t in l:
                self.assertIsInstance(t, tuple, 'Expected triples to be saved as tuples')
                self.assertEqual(3, len(t), 'Expected triples, with length 3')



class TestGetTermsDictFromList(unittest.TestCase):
    def test_equivalence(self):
        """Should contain same information after transforming to a dictionary
        """

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

        d_terms = get_terms_dict_from_list(l_terms)

        len_d_terms = sum(len(v) for v in d_terms.values())

        self.assertTrue(len(l_terms), 'Sanity check. Should be non-empty')

        self.assertEqual(len(l_terms), len_d_terms, 'No terms should be lost')


class TestGetEurovocTerms(unittest.TestCase):

    def test_instance(self):
        terms = get_eurovoc_terms()

        self.assertIsInstance(terms, dict)

    def test_fresh_download(self):
        """

        Returns:

        """

        terms_local = get_eurovoc_terms(download=False)

        with tempfile.TemporaryDirectory() as dir_temp:
            filename_temp = os.path.join(dir_temp, 'eurovoc_terms.json')
            terms_fresh = get_eurovoc_terms(download=True,
                                            filename=filename_temp)

        with self.subTest('uri'):
            for key_local in terms_local:
                self.assertIn(key_local, terms_fresh)

        with self.subTest('labels'):
            for key_local, labels_local in terms_local.items():
                self.assertEqual(set(labels_local),
                                 set(terms_fresh.get(key_local))
                                 )
