from collections import Counter
from numpy import array
from sklearn.base import BaseEstimator

__author__ = 'mmb28'


class MostCommonLabelClassifier(BaseEstimator):
    """
    A scikits classifier that always predicts the most common class
    """

    def __init__(self, decision=0):
        self.decision = decision

    def fit(self, X, y):
        c = Counter(y)
#        print '************************************'
#        print 'DumbClassifier counter = %r' % c
#        print 'Fitting: Data shape = %s' % str(X.shape)
        self.decision = c.most_common(1)[0][0]
#        print 'DumbClassifier decision = %d' % self.decision
#        print '************************************'

    def predict(self, X):
#        print '************************************'
#        print 'DumbClassifier marks %d documents as class %d' % (X.shape[0],
#                                                                 self
#                                                                 .decision)
#        print '************************************'

        return array([self.decision] * X.shape[0])

