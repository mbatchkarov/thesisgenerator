from collections import Counter

__author__ = 'mmb28'


class MostCommonLabelClassifier(object):
    """
    A scikits classifier that always predicts the most common class
    """

    def __init__(self, decision=0):
        self._decision = decision

    def fit(self, X, y):
        c = Counter(y)
        self._decision = c.most_common(1)[0][0]

    def predict(self, X):
        return [self._decision] * X.shape[0]

    #    def predict_log_proba(self, X):
    #        pass

#    def predict_proba(self, X):
#        return np.exp(self.predict_log_proba(X))

    def get_params(self, **kwargs):
        return {'decision': self._decision}
