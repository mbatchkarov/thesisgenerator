import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
from sklearn.feature_selection.univariate_selection import _clean_nans

__author__ = 'mmb28'


class VectorBackedSelectKBest(SelectKBest):
    """
    An extention of sklearn's SelectKBest, which also contains a VectorStore. Feature selection is done
    in two optional steps:
        1: Remove all features that are not contained in the vector store
        2: Remove any remaining low-scoring features to ensure a maximum of k features are left fit

     Additionally, this class stores a vocabulary (like a vectorizer), which maps features to a corresponding columns
     in the feature vector matrix. This is so that a FeatureVectorsCsvDumper can be placed after this object in a
     pipeline.

     Also, this object assumes its input is not a matrix X (as in SelectKBest), but a tuple (X, vocabulary). The
     vocabulary is provided by ThesaurusVectorizer, which comes before this object in a pipeline and represents the
     mapping of features to columns in X before any feature selection is done.
    """

    def __init__(self, score_func=chi2, k='all', vector_source={}, ensure_vectors_exist=False):
        self.k = k
        self.vector_source = vector_source
        self.ensure_vectors_exist = ensure_vectors_exist
        if not self.vector_source and ensure_vectors_exist:
            logging.error(
                'You requested feature selection based on vector presence but did not provide a vector source.')
            raise ValueError('VectorSource required with ensure_vectors_exist')
        self.vocabulary_ = None
        super(VectorBackedSelectKBest, self).__init__(score_func=score_func, k=k)

    def fit(self, X, y):
        # Vectorizer also returns its vocabulary, store it and work with the rest
        X_only, self.vocabulary_ = X

        if self.k == 'all' or int(self.k) >= X_only.shape[1]:
            # do not bother calculating feature informativeness if all features will be used anyway
            self.scores_ = np.ones((X_only.shape[1],))
        else:
            super(VectorBackedSelectKBest, self).fit(X_only, y)

        if self.ensure_vectors_exist:
            self.to_keep = self._zero_score_of_oot_features()
        return self

    def transform(self, X):
        # Vectorizer also returns its vocabulary, remove it
        return super(VectorBackedSelectKBest, self).transform(X[0]), self.vocabulary_

    def _zero_score_of_oot_features(self):
        #print 'Removing features'
        mask = np.ones(self.scores_.shape, dtype=bool)
        for feature, index in self.vocabulary_.iteritems():
            if feature not in self.vector_source:
                #print feature, 'not found'
                self.scores_[index] = 0
                mask[index] = 0
        return mask

    def _update_vocab_according_to_mask(self, mask):
        v = self.vocabulary_
        if len(v) < mask.shape[0]:
            logging.info('Already pruned %d features down to %d', mask.shape[0], len(v))
            return

        # see which features are left
        v = {feature: index for feature, index in v.iteritems() if mask[index]}
        # assign new indices for each remaining feature in order, map: old_index -> new_index
        new_indices = {old_index: new_index for new_index, old_index in enumerate(sorted(v.values()))}
        # update indices in vocabulary
        self.vocabulary_ = {feature: new_indices[index] for feature, index in v.iteritems()}

    def _get_support_mask(self):
        k = self.k
        scores = self.scores_
        if k == 'all' or k > len(scores):
            # at this point self._remove_oot_features will have been invoked, and there is no
            # further feature selection to do
            logging.warn('Using all %d features (you requested %r)', len(scores), k)
            try:
                first_mask = self.to_keep
            except AttributeError:
                # self.keep_after_first_pass_mask does not exist because self.ensure_vectors_exist=False
                # i.e. all features are kept, set a mask of ones
                first_mask = np.ones(scores.shape, dtype=bool)
            self._update_vocab_according_to_mask(first_mask)
            return scores > 0

        scores = _clean_nans(scores)
        mask = np.zeros(scores.shape, dtype=bool)
        selected_indices = np.argsort(scores)[-k:]

        mask[selected_indices] = 1
        self._update_vocab_according_to_mask(mask)
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