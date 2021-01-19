import os
import tempfile
import unittest

from fastapi.testclient import TestClient

from similar_terms.export.app import main

client = TestClient(main.app)

TERMS = ['word', 'second', 'the word', 'seconds', 'word', 'another one', 'economic', 'money', 'moneys', 'wordy',
         'public relationships', 'relationships', 'economical transactions', 'Financial transactions']
DEFS = ['', 'SeCoNd', 'the_word', '2nd', 'word', '', 'eco', 'gold', 'Plural?', '+y', 'abc', 'abc', 'abc', 'abc']


class TestExportSimTermsEurovoc(unittest.TestCase):

    def test_upload_file(self):
        # d_json = {"voc": ["word"]}

        with tempfile.TemporaryDirectory() as tmp_dir:
            filename_glossary = os.path.join(tmp_dir, 'glossary.csv')

            with open(filename_glossary, mode='w') as fp:
                delimiter = '⬤'
                s_lines = ''
                for term_i, def_i in zip(TERMS, DEFS):
                    s_lines += delimiter.join([term_i, def_i]) + '\n'

                fp.writelines(s_lines)

            with open(filename_glossary) as fp:
                files = {'file': fp}
                response = client.post("/export_sim_terms_eurovoc/",
                                       files=files,
                                       )

        with self.subTest('Status code'):
            self.assertLess(response.status_code, 300, 'Should return a valid response code')

        # TODO test content of response.
        with self.subTest('Content'):
            self.assertTrue(response.content, 'Should be non-empty')
