# coding=utf-8
from collections import Counter
from itertools import combinations
import logging
try:
    import cPickle as pickle
except ImportError:
    import pickle
from joblib import hashing
from numpy import array
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import binarize
from sklearn.utils import check_random_state

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
        return self

    def predict(self, X):
        return array([self.decision] * X.shape[0])


def score_equals_prediction(true, predicted):
    return predicted


class PicklingPipeline(Pipeline):
    """
    A pipeline extension that saves itself when it is trained
    """

    def __init__(self, steps, exp_name):
        super(PicklingPipeline, self).__init__(steps)
        self.exp_name = exp_name

    def fit(self, X, y=None, **fit_params):
        trained_pipeline = super(PicklingPipeline, self).fit(X, y, **fit_params)
        outfile_name = '{}-{}-pipeline.pickle'.format(self.exp_name, self.cv_number)
        logging.info('Saving trained pipeline {}'.format(outfile_name))

        # only save the classifier, the other stuff isn't really needed
        with open(outfile_name, 'w') as outfile:
            pickle.dump(self.named_steps['clf'], outfile)
        logging.info('Done saving')
        return trained_pipeline


class DataHashingClassifierMixin(object):
    """
    A classifier that keeps track of what is was trained and evaluated on
    and when asked to predict return a hash of that data. Multiple instances
    of this class can be used to verify if the same data is consistently
    being passed in by crossvalidation iterators with a random element
    """

    def fit(self, X, y, *args, **kwargs):
        self.train_data = X
        return self

    def fit_transform(self, X, y, *args, **kwargs):
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


class LeaveNothingOut(object):
    """A modified version of sklearn.cross_validation.LeavePOut which leaves
    nothing out, i.e. the whole dataset it used for both testing and training
    """

    def __init__(self, n, indices=True):
        self.n = n
        self.indices = indices

    def __iter__(self):
        n = self.n
        comb = combinations(range(n), n)
        for idx in comb:
            test_index = np.zeros(n, dtype=np.bool)
            test_index[np.array(idx)] = True
            #            train_index = np.logical_not(test_index)
            train_index = test_index
            if self.indices:
                ind = np.arange(n)
                train_index = ind[train_index]
                test_index = ind[test_index]
            yield train_index, test_index


class PredefinedIndicesIterator(object):
    """A scikits-compliant crossvalidation iterator which returns
    a single pair of pre-defined train-test indices. To be used when the test
     set is known in advance
    """

    def __init__(self, train, test):
        self.train = train
        self.test = test

    def __iter__(self):
        logging.info('Yielding a training set of size %d and a test set of '
                     'size %d', len(self.train), len(self.test))

        yield self.train, self.test
        raise StopIteration


class SubsamplingPredefinedIndicesIterator(object):
    """
    A CV iterator that selects a stratified sample of all available training
    documents, but returns all available test documents. Each sample contains the same ratio
    of data points from each class as the full training set. This may occasionally result in sample size
    that is slightly different that sample_size. E.g. if sample_size==5 and y_vals has the same number
    of positives and negatives, one needs to sample 4 or 6 points to preserve the 1:1 ratio
    """

    def __init__(self, y_vals, train, test, num_samples, sample_size,
                 random_state=0):
        """
        Parameters:
        :param y_vals: - all targets, for both train and test set
        :type y_vals: np.array
        :param train:- indices of the train set
        :param test:- indices of the test set
        :param num_samples:- how many CV runs to perform
        :param sample_size:- how large a sample to take from the test set
        :param random_state:- int or numpy.RandomState, as per scikit's docs
        """
        self.y_vals = y_vals
        self.train = train
        self.test = test
        self.num_samples = num_samples
        self.sample_size = int(sample_size)
        self.rng = check_random_state(random_state)
        self.counts = Counter(y_vals)
        for label, freq in self.counts.items():
            self.counts[label] /= float(len(y_vals))
        logging.info('Will do %d runs, for each sampling %d documents from a training set of size %d',
                     self.num_samples,
                     self.sample_size,
                     len(self.train))

    def __iter__(self):
        for i in range(self.num_samples):
            ind_train = np.zeros((0,), dtype=np.int)

            for label, proportion in self.counts.items():
                train_size = int(round(proportion * self.sample_size))

                ind = np.nonzero(self.y_vals[self.train] == label)[0]
                ind = self.rng.choice(ind, size=train_size, replace=False)

                logging.debug('Selected %r for class %r', ind, label)
                ind_train = np.concatenate((ind_train, ind), axis=0)
            logging.info('Will train on collection of len %r - %r', len(ind_train), sorted(ind_train))
            yield ind_train, self.test
        raise StopIteration

    def __len__(self):
        return self.num_samples


class NoopTransformer(BaseEstimator, TransformerMixin):
    """
    A no-op BaseEstimator, which does nothing to its input data
    Also functions as a joblib cache object that does nothing
    """

    def cache(self, func, **kwargs):
        return func
        # for joblib caching, act as a identity decorator

    def fit(self, X, y=None):
        return self

    def transform(self, X, copy=True):
        return X