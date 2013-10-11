import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest
import numpy as np
from sklearn.feature_selection.univariate_selection import _clean_nans

__author__ = 'mmb28'


class VectorBackedSelectKBest(SelectKBest):
    def __init__(self, score_func, k='all', vector_source={}, ensure_vectors_exist=True):
        self.k = k
        self.vector_source = vector_source
        self.ensure_vectors_exist = ensure_vectors_exist
        self.vocabulary_ = None
        super(VectorBackedSelectKBest, self).__init__(score_func=score_func, k=k)

    def fit(self, X, y):
        # Vectorizer also returns its vocabulary, store it and work with the rest
        self.vocabulary_ = X[1]
        X = X[0]

        if self.k == 'all' or int(self.k) >= X.shape[1]:
            # do not bother calculating feature informativeness if all features will be used anyway
            self.scores_ = np.ones((X.shape[1],))
        else:
            super(VectorBackedSelectKBest, self).fit(X, y)

        if self.ensure_vectors_exist:
            self.scores_ = self._remove_oot_features(self.scores_)
        return self

    def transform(self, X):
        # Vectorizer also returns its vocabulary, remove it
        return super(VectorBackedSelectKBest, self).transform(X[0])

    def _remove_oot_features(self, scores):
        #print 'Removing features'
        removed_features = set()
        v = self.vocabulary_
        for feature, index in v.items():
            if feature not in self.vector_source:
                #print feature, 'not found'
                scores[index] = 0
                removed_features.add(index)
        v = {k: v for k, v in v.iteritems() if v not in removed_features}
        new_vals = {val: pos for pos, val in enumerate(sorted(v.values()))}
        self.vocabulary_ = {key: new_vals[val] for key, val in v.iteritems()}
        return scores

    def _get_support_mask(self):
        k = self.k
        scores = self.scores_
        if k == 'all' or k > len(scores):
            # at this point self._remove_oot_features will have been invoked, and there is no
            # further feature selection to do
            logging.warn('Using all %d features (you requested %r)' % (len(scores), k))
            return scores > 0
            #if :
            #raise ValueError("Cannot select %d features among %d. "
            #                 "Use k='all' to return all features."
            #                 % (k, len(scores)))

        scores = _clean_nans(scores)
        # XXX This should be refactored; we're getting an array of indices
        # from argsort, which we transform to a mask, which we probably
        # transform back to indices later.
        mask = np.zeros(scores.shape, dtype=bool)
        mask[np.argsort(scores)[-k:]] = 1
        return mask


class MetadataStripper(BaseEstimator, TransformerMixin):
    """
    The current implementation of ThesaurusVectorizer's fit() returns not just a data matrix, but also some
    metadata (its vocabulary). This class is meant to sit in a pipeline behind the vectorizer to remove that
    metadata, so that it doesn't break other items in the pipeline.

    Currently several other pipeline elements can make use of this data ( VectorBackedSelectKBest and
     FeatureVectorsCsvDumper). This class must come after these in a pipeline as they do not have any
     defensive checks
    """

    def fit(self, X, y):
        return self

    def transform(self, X):
        # if X is a tuple, strip metadata, otherwise let it be
        return X[0] if tuple(X) == X else X