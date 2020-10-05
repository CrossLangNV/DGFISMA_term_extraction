from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def crosscheck_white_black_lists(terms_list, WHITELIST, BLACKLIST):
    """

    :param terms_list: a list of terms
    :param WHITELIST: a list of whitelisted terms
    :param BLACKLIST: a list of blacklisted terms
    :return: the rectified list of terms
    """
    for term in terms_list:
        if (term not in WHITELIST) or (term in BLACKLIST):
            terms_list.remove(term)
    return terms_list


def recalculate_tf_idf(corpus, vocabulary, MAX_LEN_NGRAM):
    """

    :param corpus: a list of text segments / <p>
    :param vocabulary: a list of terms
    :return: a dictionary with {term : tf-idf}
    """
    terms_n_tfidf = {}
    # our corpus
    data = corpus
    cv = CountVectorizer(ngram_range=(1, MAX_LEN_NGRAM+1))
    # convert text data into term-frequency matrix
    data = cv.fit_transform(data)
    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(data)
    word2tfidf = dict(zip(cv.get_feature_names(), tfidf_transformer.idf_))
    for word, score in word2tfidf.items():
        if word in vocabulary:
            terms_n_tfidf.update({word : score})
    terms_n_tfidf = {k: v for k, v in sorted(terms_n_tfidf.items(), key=lambda item: item[1], reverse=True)}
    return terms_n_tfidf

