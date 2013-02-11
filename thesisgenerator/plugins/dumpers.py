import csv
import os
import pickle
from sklearn.base import TransformerMixin

__author__ = 'mmb28'


class DatasetDumper(TransformerMixin):
    """
    Saves the vectorized input to file for inspection
    """

    def __init__(self, prefix='./'):
        self.prefix = prefix

    def fit(self, X, y=None, **fit_params):
        with open('./tmp_vocabulary', 'r') as f:
            vocabulary_ = pickle.load(f)
            os.remove('./tmp_vocabulary')
            print '*********** deleting pickled vocab file'

        new_file = self.prefix + 'PostVectorizerDump.csv'
        c = csv.writer(open(new_file, "w"))

        inverse_vocab = {index: word for (word, index) in
                         vocabulary_.iteritems()}

        v = [inverse_vocab[i] for i in range(len(inverse_vocab))]
        c.writerow(['id'] + ['target'] + v)
        from pandas import DataFrame
        rows = []
        for i in range(X.shape[0]):
            vals = ['%1.2f' % x for x in X[i, :].todense().tolist()[0]]
            c.writerow([i] + [y[i]] + vals)
            rows.append([i] + [y[i]] + vals)
        df = DataFrame(rows, columns=['id'] + ['target'] + v)
        return self

    def transform(self, X):
        return X

    def get_params(self, deep=True):
        return {}

    def set_params(self, params):
        pass