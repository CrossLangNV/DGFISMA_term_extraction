# term-extraction

use "dbuild.sh" to build the docker image

use "dcli.sh" to start a docker container

Given a document (json), e.g.: https://github.com/alina-crosslang/term-extraction/blob/master/example.json, the program will return a json containing terms and tf-idf scores for each term, e.g: 
{'european economic community': 2.5641025641025643, 'economic community': 2.521008403361345, 'council decision': 0.3361344537815126, 'declaration': 0.045078888054094664, 'president': 0.008264462809917356}

The pipeline makes use of a corpus table for the tf-idf calculation:
https://github.com/alina-crosslang/term-extraction/tree/master/media/full_dgf_jsons_table2.csv

The proper way to rank most relevant terms will eventually need to be included in the pipeline, since not all the relevant terms will have a tf-idf score, and not all highly ranked terms (especially unigrams) are equally relevant. 