from sklearn.feature_selection import SelectKBest
import numpy as np
from sklearn.feature_selection.univariate_selection import _clean_nans

__author__ = 'mmb28'


class VectorBackedSelectKBest(SelectKBest):
    def __init__(self, score_func, k='all', vector_source={}, ensure_vectors_exist=True):
        self.k = k
        self.vector_source = vector_source
        self.ensure_vectors_exist_at_decode_time = ensure_vectors_exist
        self.ensure_vectors_exist = False
        self.vocabulary_ = None
        super(VectorBackedSelectKBest, self).__init__(score_func=score_func, k=k)

    def fit_transform(self, X, y):
        # Vectorizer also returns its vocabulary, store it and work with the rest
        self.vocabulary_ = X[1]
        X = X[0]

        #self.scores_, self.pvalues_ = self.score_func(X, y)
        #self.scores_ = np.asarray(self.scores_)
        #self.pvalues_ = np.asarray(self.pvalues_)
        #if len(np.unique(self.scores_)) < len(self.scores_):
        #    warn("Duplicate scores. Result may depend on feature ordering."
        #         "There are probably duplicate features, or you used a "
        #         "classification score for a regression task.")
        #
        #
        #
        #print self.scores_, self.pvalues_ > 0
        blah = super(VectorBackedSelectKBest, self).fit(X, y)
        self.ensure_vectors_exist = self.ensure_vectors_exist_at_decode_time
        return X


    def transform(self, X):
        # Vectorizer also returns its vocabulary, remove it
        return super(VectorBackedSelectKBest, self).transform(X[0])

    def remove_oot_features(self, scores):
        print 'Removing features'
        for feature, index in self.vocabulary_.items():
            if feature not in self.vector_source:
                print feature, 'not found'
                scores[index] = 0
        return scores

    def _get_support_mask(self):
        k = self.k
        if self.ensure_vectors_exist:
            scores = self.remove_oot_features(self.scores_)
        if k == 'all':
            return scores > 0
        if k > len(scores):
            raise ValueError("Cannot select %d features among %d. "
                             "Use k='all' to return all features."
                             % (k, len(self.scores_)))

        scores = _clean_nans(scores)
        # XXX This should be refactored; we're getting an array of indices
        # from argsort, which we transform to a mask, which we probably
        # transform back to indices later.
        mask = np.zeros(scores.shape, dtype=bool)
        mask[np.argsort(scores)[-k:]] = 1
        return mask