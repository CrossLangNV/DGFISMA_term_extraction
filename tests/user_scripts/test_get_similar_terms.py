import unittest
from tempfile import NamedTemporaryFile

import plac

from user_scripts import get_similar_terms


def get_temp_term(term='decisions and the assessment of fit'):
    tmp = NamedTemporaryFile(suffix='.txt')

    with open(tmp.name, 'w') as f:
        f.writelines(term)

    return tmp


def get_temp_terms_voc():
    l = ['banks',
         'board',
         'decision',
         'decisions',
         'heads',
         'work units',
         ]

    tmp = NamedTemporaryFile(suffix='.txt')
    with open(tmp.name, 'w') as f:
        f.writelines('\n'.join(l))

    return tmp


class TestMain(unittest.TestCase):
    def test_main(self):
        with get_temp_term() as f_term, \
                get_temp_terms_voc() as f_terms_voc, \
                NamedTemporaryFile(suffix='.txt') as f_out:
            self.assertIsNotNone(get_similar_terms.main(f_term.name, f_terms_voc.name, f_out.name))

    def test_cmd(self):
        with get_temp_term() as f_term, \
                get_temp_terms_voc() as f_terms_voc, \
                NamedTemporaryFile(suffix='.txt') as f_out:
            arglist = [f_term.name, f_terms_voc.name, f_out.name]

            plac.call(get_similar_terms.main, arglist=arglist)

            with open(f_out.name) as f:
                a = f.read().splitlines()

            self.assertTrue(a, 'should be non-empty')

    def test_help(self):
        arglist = ['--help']

        plac.call(get_similar_terms.main, arglist=arglist)


if __name__ == '__main__':
    unittest.main()
