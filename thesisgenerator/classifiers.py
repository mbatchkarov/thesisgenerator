from collections import Counter
from numpy import array
from sklearn.base import BaseEstimator

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

