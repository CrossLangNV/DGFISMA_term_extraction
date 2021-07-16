import os
import time

from user_scripts.similar_terms import export_glossary_link_eurovoc_rdf

ROOT = os.path.join(os.path.dirname(__file__), '../..')

FILENAME_CONCEPTS = os.path.abspath(os.path.join(ROOT, 'media/tmp/concepts_2021_07_15.csv'))

# Only saves to Turtle store, to save time.
FILENAME_RDF = os.path.abspath(os.path.join(ROOT, 'media/tmp/example_export_glossary.ttl'))

def main():
    start_time = time.time()

    a = export_glossary_link_eurovoc_rdf.main(FILENAME_CONCEPTS,
                                              filename_rdf=FILENAME_RDF,
                                              b_sim_self=True)

    end_time = time.time()
    time_elapsed = (end_time - start_time)

    print('Total elapsed time:', time_elapsed)
    return a


if __name__ == '__main__':
    main()
