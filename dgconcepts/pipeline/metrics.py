from sklearn.feature_extraction.text import TfidfVectorizer
import operator

def calculate_tf_idf(corpus, MAX_LEN_NGRAM, term_list):
    """
    Function to calculate tf_idf_score.

    :param corpus: a list of text segments / <p>
    :param vocabulary: a list of terms
    :return: {term : tf-idf}
    """

    vectorizer = TfidfVectorizer(vocabulary=term_list,ngram_range=(1, MAX_LEN_NGRAM+1), sublinear_tf=True)
    vectorizer.fit_transform(corpus)
    terms_n_tfidf = {}
    for term, score in zip(vectorizer.get_feature_names(), vectorizer.idf_):
        terms_n_tfidf.update({term : score})

    terms_n_tfidf = dict(sorted(terms_n_tfidf.items(), key=operator.itemgetter(1), reverse=True))

    return terms_n_tfidf