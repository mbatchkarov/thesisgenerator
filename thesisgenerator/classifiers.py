# coding=utf-8
from collections import Counter
from joblib import hashing
from numpy import array
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import binarize

__author__ = 'mmb28'


class MostCommonLabelClassifier(BaseEstimator):
    """
    A scikit-learn compliant classifier that always predicts the most common
    class
    """

    def __init__(self, decision=0):
        self.decision = decision

    def fit(self, X, y):
        c = Counter(y)
        self.decision = c.most_common(1)[0][0]

    def predict(self, X):
        return array([self.decision] * X.shape[0])


def score_equals_prediction(true, predicted):
    return predicted


class DataHashingClassifierMixin(object):
    """
    A classifier that keeps track of what is was trained and evaluated on
    and when asked to predict return a hash of that data. Multiple instances
    of this class can be used to verify if the same data is consistently
    being passed in by crossvalidation iterators with a random element
    """

    def fit(self, X, y, sample_weight=None, class_prior=None):
        self.train_data = X
        return self

    def fit_transform(self, X, y, sample_weight=None, class_prior=None):
        self.train_data = X
        return self

    def predict(self, X):
        h1 = hashing.hash(X.todense())
        h1 = ''.join([str(ord(x)) for x in h1])

        h2 = hashing.hash(self.train_data.todense())
        h2 = ''.join([str(ord(x)) for x in h2])

        return hash(h1 + h2)


class DataHashingNaiveBayes(DataHashingClassifierMixin, MultinomialNB):
    """
    Dummy class, the current pipeline can only work with one instance of a
    classifier type, so to get two identical classifiers we need to create
    two different subclasses
    """
    pass


class DataHashingLR(DataHashingClassifierMixin, LogisticRegression):
    pass


class MultinomialNBWithBinaryFeatures(MultinomialNB):
    """
    A Multinomial Naive Bayes with binary features, as described in
    Metsis et al (2006)
    """

    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None,
                 threshold=0.):
        self.threshold = threshold
        super(MultinomialNBWithBinaryFeatures, self).__init__(alpha, fit_prior,
                                                              class_prior)

    def fit(self, X, y, sample_weight=None, class_prior=None):
        X = binarize(X, threshold=self.threshold)
        return super(MultinomialNBWithBinaryFeatures, self).fit(X, y,
                                                                sample_weight,
                                                                class_prior)

    def predict(self, X):
        X = binarize(X, threshold=self.threshold)
        return super(MultinomialNBWithBinaryFeatures, self).predict(X)

