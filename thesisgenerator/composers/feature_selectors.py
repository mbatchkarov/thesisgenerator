from sklearn.feature_selection import SelectKBest
import numpy as np
from sklearn.feature_selection.univariate_selection import _clean_nans

__author__ = 'mmb28'


class VectorBackedSelectKBest(SelectKBest):
    def __init__(self, score_func, k='all', vector_source={}, ensure_vectors_exist=True):
        self.k = k
        self.vector_source = vector_source
        #self.ensure_vectors_exist_at_decode_time = ensure_vectors_exist
        self.ensure_vectors_exist = ensure_vectors_exist
        #self.ensure_vectors_exist = False
        self.vocabulary_ = None
        super(VectorBackedSelectKBest, self).__init__(score_func=score_func, k=k)

    def fit(self, X, y):
        # Vectorizer also returns its vocabulary, store it and work with the rest
        self.vocabulary_ = X[1]
        X = X[0]

        super(VectorBackedSelectKBest, self).fit(X, y)
        if self.ensure_vectors_exist:
            self.scores_ = self._remove_oot_features(self.scores_)
        return self

    def transform(self, X):
        # Vectorizer also returns its vocabulary, remove it
        return super(VectorBackedSelectKBest, self).transform(X[0])

    def _remove_oot_features(self, scores):
        print 'Removing features'
        removed_features = set()
        v = self.vocabulary_
        for feature, index in v.items():
            if feature not in self.vector_source:
                print feature, 'not found'
                scores[index] = 0
                removed_features.add(index)
        v = {k: v for k, v in v.iteritems() if v not in removed_features}
        new_vals = {val: pos for pos, val in enumerate(sorted(v.values()))}
        self.vocabulary_ = {key: new_vals[val] for key, val in v.iteritems()}
        return scores

    def _get_support_mask(self):
        k = self.k
        scores = self.scores_
        if k == 'all':
            # at this point self._remove_oot_features will have been invoked
            return scores > 0
        if k > len(scores):
            raise ValueError("Cannot select %d features among %d. "
                             "Use k='all' to return all features."
                             % (k, len(scores)))

        scores = _clean_nans(scores)
        # XXX This should be refactored; we're getting an array of indices
        # from argsort, which we transform to a mask, which we probably
        # transform back to indices later.
        mask = np.zeros(scores.shape, dtype=bool)
        mask[np.argsort(scores)[-k:]] = 1
        return mask