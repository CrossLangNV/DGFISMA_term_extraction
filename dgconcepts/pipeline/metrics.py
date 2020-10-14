from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def crosscheck_white_black_lists(terms_list, WHITELIST, BLACKLIST):
    """

    :param terms_list: a list of terms
    :param WHITELIST: a list of whitelisted terms
    :param BLACKLIST: a list of blacklisted terms
    :return: the rectified list of terms
    """
    clean_term_list = WHITELIST
    for term in terms_list:
        if term not in set(BLACKLIST):
            clean_term_list.append(term)
    return clean_term_list

def calculate_tf_idf(corpus, MAX_LEN_NGRAM, term_list):
    """

    :param corpus: a list of text segments / <p>
    :param vocabulary: a list of terms
    :return: {term : tf-idf}
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    import operator

    vectorizer = TfidfVectorizer(vocabulary=term_list,ngram_range=(1, MAX_LEN_NGRAM+1), sublinear_tf=True)
    vectorizer.fit_transform(corpus)
    terms_n_tfidf = {}
    for term, score in zip(vectorizer.get_feature_names(), vectorizer.idf_):
        terms_n_tfidf.update({term : score})

    terms_n_tfidf = dict(sorted(terms_n_tfidf.items(), key=operator.itemgetter(1), reverse=True))


    """
    cv = CountVectorizer(vocabulary=term_list,ngram_range=(1, MAX_LEN_NGRAM+1))
    corpus = cv.fit_transform(corpus)
    tfidf_transformer = TfidfTransformer().fit_transform(corpus)  # tf-idf matrix
    sums = tfidf_transformer.sum(axis=0)
    data = []
    for col, term in enumerate(term_list):
        data.append((term, sums[0, col]))
    import pandas as pd
    ranking = pd.DataFrame(data, columns=['term', 'rank'])
    print(ranking.sort_values('rank', ascending=False))
    terms_n_tfidf1 = {}
    for term, score in zip(ranking['term'], ranking['rank']):
        terms_n_tfidf1.update({term : score})
    import operator
    terms_n_tfidf1 = dict(sorted(terms_n_tfidf1.items(), key=operator.itemgetter(1), reverse=True))

    terms_n_tfidf2 = dict(zip(cv.get_feature_names(), tfidf_transformer.idf_))
    terms_n_tfidf2 = dict(sorted(terms_n_tfidf2.items(), key=operator.itemgetter(1), reverse=True))

    print(terms_n_tfidf1)
    print('\nLA\nLA\n')
    print(terms_n_tfidf2)
    """
    return terms_n_tfidf