# coding=utf-8

"""
A collection of random useful utilities
"""
import fileinput
import inspect
from itertools import combinations
import logging
import re
import sys
import os.path as p
import itertools
from configobj import ConfigObj
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils import check_random_state
import numpy as np


try:
    from xml.etree import cElementTree as ET
except ImportError:
    from xml.etree import ElementTree as ET


def get_susx_mysql_conn():
    import MySQLdb as mdb

    config = ConfigObj('thesisgenerator/db-credentials')
    if not config:
        return None
        # thesisgenerator/db-credentials file not found. This is needed for a
        # MySQL connection

    return mdb.connect(config['server'],
                       config['user'],
                       config['pass'],
                       config['db'])


def get_confrc(conf_file):
    """
    Searches the file hierarchy top to bottom for confrc,
    starting from  conf_file and going as many as 3 levels up
    """
    for i in range(4):
        my_list = itertools.chain([p.dirname(conf_file)],
                                  ['..'] * i,
                                  ['confrc'])
        candidate = p.join(*[x for x in my_list])
        if p.exists(candidate):
            return candidate


def get_named_object(pathspec):
    """Return a named from a module.
    """
    parts = pathspec.split('.')
    module = ".".join(parts[:-1])
    mod = __import__(module, fromlist=parts[-1])
    named_obj = getattr(mod, parts[-1])
    return named_obj


def replace_in_file(file_name, search_exp, replace_exp):
    fh = fileinput.input(file_name, inplace=1)
    for line in fh:
        line = re.sub(search_exp, replace_exp, line)
        sys.stdout.write(line)
    fh.close()


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
        logging.getLogger('root').info('Yielding a training set of '
                                       'size %d and a test set of '
                                       'size %d' %
                                       (len(self.train),
                                        len(self.test)))

        yield self.train, self.test
        raise StopIteration


class SubsamplingPredefinedIndicesIterator(object):
    """
    A CV iterator that selects a stratified sample of all available training
    documents, but returns all available test documents
    """

    def __init__(self, y_vals, train, test, num_samples, sample_size,
                 random_state=0):
        """
        Parameters:
        y_vals - all targets, for both train and test set
        train/test- indices of the train/test set
        num_sample- how many CV runs to perform
        sample_size- how large a sample to take from the test set
        random_state- int or numpy.RandomState, as per scikit's docs
        """
        self.y_vals = y_vals
        self.train = train
        self.test = test
        self.num_samples = num_samples
        self.sample_size = int(sample_size)
        self.rng = check_random_state(random_state)
        self.counts = np.bincount(y_vals) / float(len(y_vals))
        logging.getLogger('root').info('Will do %d runs, '
                                       'for each sampling %d documents from a '
                                       'training set of size %d' % (
                                           self.num_samples,
                                           self.sample_size,
                                           len(self.train)))

    def __iter__(self):
        for i in range(self.num_samples):
            ind_train = np.zeros((0,), dtype=np.int)

            for label, proportion in enumerate(list(self.counts)):
                train_size = int(round(proportion * self.sample_size))

                ind = np.nonzero(self.y_vals[self.train] == label)[0]
                ind = self.rng.choice(ind, size=train_size, replace=False)

                logging.getLogger('root').debug(
                    'Selected %r for class %r' % (ind, label))
                ind_train = np.concatenate((ind_train, ind), axis=0)
            logging.getLogger('root').info(
                'Will train on collection of len %r - %r' % (
                    len(ind_train), ind_train))
            yield ind_train, self.test
        raise StopIteration

    def __len__(self):
        return self.num_samples


class NoopTransformer(BaseEstimator, TransformerMixin):
    """
    A no-op base estimator, which does nothing to its input data
    Also functions as a joblib cache object that does nothing
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, copy=True):
        return X

    def cache(self, x):
        return x


class ChainCallable(object):
    """
    Chains several functions as specified in the configuration. When called
    returns the return values of all of its callables as a tuple

    Example usage:

    # some sample functions
    def t1(shared_param, conf):
        print 'called t1(%s, %s)' % (str(shared_param), str(shared_param))
        return 1


    def t2(shared_param, conf):
        print 'called t1(%s, %s)' % (str(shared_param), str(shared_param))
        return 2

    config = {'a': 1, 'b': 2}
    ccall = chain_callable(config)
    print ccall('X_Y')
    print ccall('A_B')

    """

    def __init__(self, config):
        self.to_call = [(x, get_named_object(x)) for x in
                        config.keys() if config[x]['run']]
        self.config = config

    def __call__(self, true_labels, predicted_labels):
        options = {}
        result = {}
        for func_name, func in self.to_call:
            initialize_args = inspect.getargspec(func)[0]
            call_args = {arg: val for arg, val in self.config[func_name].items()
                         if val != '' and arg in initialize_args}
            options[func_name] = call_args
            result[func_name.strip()] = (
                func(true_labels, predicted_labels, **call_args))
        return result
