from pipeline import terms
import pandas as pd
import os
import sys
import jsonlines

data_dir = sys.argv[1]
output_name = os.path.join(os.getcwd(), 'dgf-voc.csv')
if os.path.exists(output_name):
    corpus_table = pd.read_csv(output_name)
else:
    corpus_table = pd.DataFrame(columns=["ngrams"])

for file in os.listdir(data_dir):
    reader = jsonlines.open(os.path.join(data_dir, file))
    for line in reader:
        try:
            sentences = line['content']
            dict_v1, abvs = terms.analyzeFile(''.join(sentences))
            voc_filtered = pd.DataFrame(dict_v1['ngrams'], columns=['ngrams'])
            corpus_table = corpus_table.append(voc_filtered)
            corpus_table.to_csv(output_name)
        except:
            continue