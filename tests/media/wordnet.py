import unittest

from media.wordnet import wordnet_concepts, wordnet_related_concepts


class TestWordNetConcepts(unittest.TestCase):
    def test_return(self):
        concepts = wordnet_concepts()

        with self.subTest("type"):
            self.assertIsInstance(concepts, dict)
        with self.subTest('type elements'):
            for l_labels in concepts.values():
                self.assertIsInstance(l_labels, list)


class TestWordNetRelatedConcepts(unittest.TestCase):
    def test_return(self):
        related_concepts = wordnet_related_concepts()

        with self.subTest("type"):
            self.assertIsInstance(related_concepts, dict)

        with self.subTest('type elements'):
            for d_concepts_relation in related_concepts.values():
                self.assertIsInstance(d_concepts_relation, dict)
