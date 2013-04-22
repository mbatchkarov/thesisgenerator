from collections import defaultdict
import csv
import os
import pickle
from numpy import count_nonzero
from sklearn.base import TransformerMixin

__author__ = 'mmb28'


class DatasetDumper(TransformerMixin):
    """
    Saves the vectorized input to file for inspection
    """

    def __init__(self, pipe_id=0, prefix='.'):
        self.pipe_id = pipe_id
        self.prefix = prefix
        self._tranform_call_count = 0

    def _dump(self, X, y, file_name='dump.csv'):
        """
        The call order is
            1. training data
                1. fit
                2. transform
            2. test data
                1. transform

        We only want to dump after stages 1.2 and 2.1
        """
        vocab_file = './tmp_vocabulary%d' % self.pipe_id
        with open(vocab_file, 'r') as f:
            vocabulary_ = pickle.load(f)
            if len(y) < 1:
                print '*********** deleting pickled vocab file'
                os.remove(vocab_file)
        new_file = os.path.join(self.prefix, file_name)
        c = csv.writer(open(new_file, "w"))
        inverse_vocab = {index: word for (word, index) in
                         vocabulary_.iteritems()}
        v = [inverse_vocab[i] for i in range(len(inverse_vocab))]
        c.writerow(['id'] + ['target'] + ['total_feat_weight'] +
                   ['nonzero_feats'] + v)
        for i in range(X.shape[0]):
            row = X.todense()[i, :].tolist()[0]
            vals = ['%1.2f' % x for x in row]
            c.writerow([i, y[i], sum(row), count_nonzero(row)] + vals)

    def fit(self, X, y=None, **fit_params):
        self.y = y
        return self

    def transform(self, X):
        self._tranform_call_count += 1
        suffix = {1: 'tr', 2: 'ev'}
        if not hasattr(self, 'y'):
            self.y = defaultdict(str)

        if 1 <= self._tranform_call_count <= 2:
            self._dump(X, self.y, file_name='PostVectDump-%s%d.csv' % (
                suffix[self._tranform_call_count],
                self.pipe_id))
        return X

    def get_params(self, deep=True):
        return {'pipe_id': self.pipe_id}

    def set_params(self, params):
        pass