import pandas as pd


def crosscheck_unigrams(term_dict):

    """
    Zipf curve might be useful for sorting the terms
    """
    unigrams = []
    rest = []
    for x in term_dict:
        if len(x.split()) == 1:
            unigrams.append(x)
        else:
            rest.append(x)
    grr = ' '.join(rest).split()
    for x in unigrams:
        if x not in grr:
            term_dict.pop(x, None)
    return term_dict


def check_abbvs(dict_1, abbvs):
    """
    for x in dict_1['ngrams']:
        if x in list(abbvs['abbv']):
            dict_1['ngrams'][dict_1['ngrams'].index(x)] = x + "■" + abbvs['term'][abbvs['abbv'].index(x)]
        if x in list(abbvs['term']):
            dict_1['ngrams'][dict_1['ngrams'].index(x)] = abbvs['abbv'][abbvs['term'].index(x)] + "■" + x
    """
    return dict_1


def update_voc(dict_1, PATH_TO_VOC):
    voc = pd.read_csv(PATH_TO_VOC)
    raw_terms_and_counts = list(zip(dict_1['ngrams'], dict_1['count']))
    raw_terms_and_counts = pd.DataFrame(raw_terms_and_counts, columns=['ngrams', 'count']).sort_values(by='count',
                                                                                                       ascending=False)
    aggregation_functions = {'count': 'sum'}
    voc = voc.append(raw_terms_and_counts)
    new_voc = voc.groupby(voc['ngrams']).aggregate(aggregation_functions)
    new_voc = new_voc.sort_values(by='df', ascending=False)
    new_voc.to_csv(PATH_TO_VOC)


def update_pd2(dict_1, PATH_TO_PD2):  # explain what is pd2
    pandas_dataframe_2 = pd.read_csv(PATH_TO_PD2)
    pandas_dataframe_1 = pd.DataFrame.from_dict(dict_1)
    corpus_table = pandas_dataframe_2.append(pandas_dataframe_1)
    aggregation_functions = {'df': 'sum'}
    corpus_table = corpus_table.groupby(corpus_table['ngrams']).aggregate(aggregation_functions)
    corpus_table = corpus_table.sort_values(by='df', ascending=False)
    corpus_table.to_csv(PATH_TO_PD2)


def calculate_tf_idf(dict_1, PATH_TO_PD2):  # number of docs meegeven
    pandas_dataframe_1 = pd.DataFrame.from_dict(dict_1)
    pandas_dataframe_2 = pd.read_csv(PATH_TO_PD2)
    t1 = pd.merge(pandas_dataframe_1, pandas_dataframe_2, how='left', on=['ngrams'])
    t1.df = t1.df + 1

    # TO DO : avoid the magic number
    idf = 240 / t1.df  # 240 is number of docs from which pd2 was created, 10320 for europarl
    t1['idf'] = idf
    t1['tfidf'] = t1.tf * t1.idf
    t1 = t1.fillna(0)
    t1 = t1.sort_values(by='tfidf', ascending=False)
    terms_n_tfidf = {}
    for x, y in zip(t1['ngrams'], t1['tfidf']):
        terms_n_tfidf.update({x: y})
    return terms_n_tfidf
