# coding=utf-8
from collections import defaultdict
import csv
import logging
from operator import itemgetter
import os
import pickle
import itertools
from numpy import count_nonzero
from sklearn.base import TransformerMixin

__author__ = 'mmb28'


class FeatureVectorsCsvDumper(TransformerMixin):
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
        if self._tranform_call_count == 2:
            self.y = defaultdict(str)

        if 1 <= self._tranform_call_count <= 2:
            self._dump(X, self.y, file_name='PostVectDump-%s%d.csv' % (
                suffix[self._tranform_call_count],
                self.pipe_id))
        return X

    def get_params(self, deep=True):
        return {'pipe_id': self.pipe_id}


columns = [('name', 'TEXT'),
           ('sample_size', 'INTEGER'),
           ('train_voc_mean', 'INTEGER'),
           ('train_voc_std', 'INTEGER'),
           ('corpus', 'INTEGER'),
           ('features', 'INTEGER'),
           ('pos', 'TEXT'),
           ('fef', 'INTEGER'),
           ('classifier', 'TEXT'),
           ('th_size', 'INTEGER'),
           ('total_tok', 'INTEGER'),
           ('unknown_tok_mean', 'INTEGER'),
           ('unknown_tok_std', 'INTEGER'),
           ('found_tok_mean', 'INTEGER'),
           ('found_tok_std', 'INTEGER'),
           ('replaced_tok_mean', 'INTEGER'),
           ('replaced_tok_std', 'INTEGER'),
           ('total_typ', 'INTEGER'),
           ('unknown_typ_mean', 'INTEGER'),
           ('unknown_typ_std', 'INTEGER'),
           ('found_typ_mean', 'INTEGER'),
           ('found_typ_std', 'INTEGER'),
           ('replaced_typ_mean', 'INTEGER'),
           ('replaced_typ_std', 'INTEGER'),
           ('metric', 'TEXT'),
           ('score_mean', 'FLOAT'),
           ('score_std', 'FLOAT')]


class ConsolidatedResultsCsvWriter(object):
    def __init__(self, output_fh):
        self.c = csv.writer(output_fh)
        self.c.writerow([x[0] for x in columns])

    def writerow(self, row):
        self.c.writerow(row)

    def __str__(self):
        return 'ConsolidatedResultsCsvWriter-%s' % self.c


class ConsolidatedResultsSqlWriter(object):
    def __init__(self, table_number, output_db_conn):
        self.conn = output_db_conn
        c = self.conn.cursor()

        self.table_name = 'data%.2d' % table_number

        c.execute('DROP TABLE IF EXISTS %s' % self.table_name)
        params = [item for col in columns for item in col]
        template = 'CREATE TABLE %s(' + \
                   ', '.join(['%s %s'] * (len(params) / 2)) + \
                   ')'
        q = template % tuple(itertools.chain([self.table_name], params))
        c.execute(q)
        self.conn.commit()

    def writerow(self, row):
        template = "INSERT INTO %s(" + \
                   ', '.join(['%s'] * len(columns)) + \
                   ") VALUES (" + \
                   ', '.join(['%s'] * len(row)) + \
                   ")"
        sql = template % tuple(itertools.chain([self.table_name],
                                               map(itemgetter(0), columns),
                                               row))
        self.conn.cursor().execute(sql)
        self.conn.commit()

    def __del__(self):
        self.conn.commit()
        self.conn.close()

    def __str__(self):
        return 'ConsolidatedResultsSqliteWriter-%s' % self.table_name


class DummySqlWriter(object):
    """
    A null SqlWriter object
    """

    def writerow(self, row):
        pass


class ConsolidatedResultsSqlAndCsvWriter(object):
    def __init__(self, table_number, csv_output_fh, output_db_conn):
        self.csv = ConsolidatedResultsCsvWriter(csv_output_fh)
        if output_db_conn:
            self.sql_conn = ConsolidatedResultsSqlWriter(table_number,
                                                         output_db_conn)
        else:
            logging.getLogger('root').warn("Database connection impossible")
            self.sql_conn = DummySqlWriter()

    def writerow(self, row):
        self.csv.writerow(row)
        self.sql_conn.writerow(row)

    def __str__(self):
        return 'ConsolidatedResultsSqliteAndCsvWriter-%s-%s' % (
            self.csv, self.sql_conn)
