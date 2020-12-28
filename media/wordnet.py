"""
Open Multilingual Wordnet
From http://compling.hss.ntu.edu.sg/omw/
"""

import json
import os
from typing import List

import nltk
from nltk.corpus import wordnet as w
from nltk.corpus.reader.wordnet import Synset

b = False
if b:
    # Downloader
    nltk.download()
    # - press d) Download
    # - Insert "wordnet"


def wordnet_concepts(download=False,
                     filename=os.path.join(os.path.dirname(__file__), 'wordnet_terms.json'),
                     ) -> List[str]:
    if download or not os.path.exists(filename):

        d_terms = {}
        for synset in w.all_synsets():

            s_id = _wordnet_id(synset)

            for lemma in synset.lemmas():
                label = lemma.name().replace('_', ' ')

                d_terms.setdefault(s_id, []).append(label)

        with open(filename, 'w') as outfile:
            json.dump(d_terms, outfile, indent=4)

        del (d_terms)

    with open(filename, 'r') as json_file:
        d_terms = json.load(json_file)

    return d_terms


def wordnet_related_concepts(download=False,
                     filename=os.path.join(os.path.dirname(__file__), 'wordnet_related_concepts.json'),
                     ):

    if download or not os.path.exists(filename):

        d_related_concepts = {}

        n_synset = 0
        n_lemma = 0
        n_hypernym = 0
        n_hyponym = 0

        for synset in w.all_synsets():
            n_synset += 1
            for lemma in synset.lemmas():

                n_lemma += 1

                # Do nothing with Lemma's

            for hypernym in synset.hypernyms():
                n_hypernym += 1

                d_related_concepts.setdefault(_wordnet_id(synset), {})[_wordnet_id(hypernym)] = 'hypernym'

            for hyponym in synset.hyponyms():
                n_hyponym += 1

                d_related_concepts.setdefault(_wordnet_id(synset), {})[_wordnet_id(hyponym)] = 'hyponym'

        with open(filename, 'w') as outfile:
            json.dump(d_related_concepts, outfile, indent=4)

        del (d_related_concepts)

    with open(filename, 'r') as json_file:
        d_related_concepts = json.load(json_file)

    return d_related_concepts


def _wordnet_id(synset: Synset) -> str:
    s_id = "%08d-%s" % (synset.offset(), synset.pos())
    return s_id
