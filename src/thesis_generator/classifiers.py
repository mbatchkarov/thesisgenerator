from collections import Counter
from sklearn.base import BaseEstimator

__author__ = 'mmb28'


class MostCommonLabelClassifier(BaseEstimator):
    """
    A scikits classifier that always predicts the most common class
    """

    def __init__(self, decision=0):
        self._decision = decision

    def fit(self, X, y):
        c = Counter(y)
        print '************************************'
        print 'DumbClassifier counter =%r', c
        self._decision = c.most_common(1)[0][0]
        print 'DumbClassifier decision =%d', self._decision
        print '************************************'

    def predict(self, X):
        print '************************************'
        print 'DumbClassifier marks %d documents as class %d'%(X.shape[0],
                                                               self._decision)
        print '************************************'

        return [self._decision] * X.shape[0]

    def get_params(self, deep=True):
        x = super(MostCommonLabelClassifier, self).get_params(deep)
        x['decision'] = self._decision
        return x
