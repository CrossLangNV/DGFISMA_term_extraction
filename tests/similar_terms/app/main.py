import unittest

from fastapi.testclient import TestClient

from similar_terms.app import main

client = TestClient(main.app)


class TestReadMain(unittest.TestCase):
    def test_read_main(self):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"msg": "Similar term retrieval"}


class TestAlignVocs(unittest.TestCase):
    def test_load_two_vocs(self):
        d_json = {"voc1": ["word 1", "another word"],
                  "voc2": ["word 1", "word 2"]}

        response = client.post("/similar_terms/align/", json=d_json)

        self.assertEqual(200, response.status_code)

        json_return = response.json()

        self.assertIsInstance(json_return, dict)

        with self.subTest('keys class'):
            for k in json_return.keys():
                self.assertIsInstance(k, str)

        with self.subTest('values class'):
            for v in json_return.values():
                self.assertIsInstance(v, list)

                for l in v:
                    self.assertIsInstance(l, str)

    def test_identical(self):
        d_json = {'voc1': ['word'],
                  'voc2': ['word']}

        response = client.post("/similar_terms/align/", json=d_json)

        self.assertEqual(response.json(), {'word': ['word']})


class TestAlignWithSelf(unittest.TestCase):
    def test_identical(self):
        d_json = {"voc": ["word"]}

        response = client.post("/similar_terms/self/", json=d_json)

        self.assertEqual(response.json(), {'word': ['word']})


if __name__ == '__main__':
    unittest.main()
