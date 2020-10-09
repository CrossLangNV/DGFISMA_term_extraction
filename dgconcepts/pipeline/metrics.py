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
    cv = CountVectorizer(vocabulary=term_list,ngram_range=(1, MAX_LEN_NGRAM+1))
    corpus = cv.fit_transform(corpus)
    tfidf_transformer = TfidfTransformer()
    tfidf_transformer.fit_transform(corpus)  # tf-idf matrix
    terms_n_tfidf = dict(zip(cv.get_feature_names(), tfidf_transformer.idf_))
    return terms_n_tfidf