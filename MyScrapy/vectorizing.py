from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_vectorizing(data):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(data).toarray()
    return tfidf, tfidf_matrix


def count_vectorizing(data):
    count = CountVectorizer(stop_words="english")
    count_matrix = count.fit_transform(data).toarray()
    return count, count_matrix
