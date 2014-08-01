# coding=utf-8
from collections import defaultdict
import csv
import logging
from operator import itemgetter
import os
import itertools
import platform
from numpy import count_nonzero
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
from sklearn.base import TransformerMixin
from thesisgenerator.plugins.consolidator import consolidate_results
from thesisgenerator.utils.misc import get_susx_mysql_conn

__author__ = 'mmb28'


class FeatureVectorsCsvDumper(TransformerMixin):
    """
    Saves the vectorized input to file for inspection
    """

    def __init__(self, exp_name, cv_number=0, prefix='.'):
        self.exp_name = exp_name
        self.cv_number = cv_number
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
        matrix, vocabulary_ = X
        new_file = os.path.join(self.prefix, file_name)
        c = csv.writer(open(new_file, "w"))
        inverse_vocab = {index: word for (word, index) in vocabulary_.items()}
        v = [inverse_vocab[i] for i in range(len(inverse_vocab))]
        c.writerow(['id'] + ['target'] + ['total_feat_weight'] + ['nonzero_feats'] + v)
        for i in range(matrix.shape[0]):
            row = matrix.todense()[i, :].tolist()[0]
            vals = ['%1.2f' % x for x in row]
            c.writerow([i, y[i], sum(row), count_nonzero(row)] + vals)
        logging.info('Saved debug info to %s', new_file)

    def fit(self, X, y=None, **fit_params):
        self.y = y
        return self

    def transform(self, X):
        self._tranform_call_count += 1
        suffix = {1: 'tr', 2: 'ev'}
        if self._tranform_call_count == 2:
            self.y = defaultdict(str)

        if 1 <= self._tranform_call_count <= 2:
            self._dump(X, self.y,
                       file_name='PostVectDump_%s_%s-fold%r.csv' % (self.exp_name,
                                                                     suffix[self._tranform_call_count],
                                                                     self.cv_number))
        return X

    def get_params(self, deep=True):
        return {'cv_number': self.cv_number,
                'exp_name': self.exp_name}


columns = [('id', 'INTEGER NOT NULL AUTO_INCREMENT'),
           ('name', 'TEXT'),
           ('git_hash', 'TEXT'),
           ('consolidation_date', 'TIMESTAMP'),

           # experiment settings
           ('cv_folds', 'INTEGER'),
           ('sample_size', 'INTEGER'),
           ('classifier', 'TEXT'),

           #  performance
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

        self.table_name = 'data%d' % table_number

        c.execute('DROP TABLE IF EXISTS %s' % self.table_name)
        params = [item for col in columns for item in col]
        template = 'CREATE TABLE %s(' + \
                   ', '.join(['%s %s'] * (len(params) // 2)) + \
                   ', PRIMARY KEY(id))'
        q = template % tuple(itertools.chain([self.table_name], params))
        c.execute(q)
        self.conn.commit()

    def writerow(self, row):
        template = "INSERT INTO %s(" + \
                   ', '.join(['%s'] * len(columns)) + \
                   ") VALUES (" + \
                   ', '.join(['\"%s\"'] * len(row)) + \
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
            logging.warning("Database connection impossible")
            self.sql_conn = DummySqlWriter()

    def writerow(self, row):
        self.csv.writerow(row)
        self.sql_conn.writerow(row)

    def __str__(self):
        return 'ConsolidatedResultsSqliteAndCsvWriter-%s-%s' % (
            self.csv, self.sql_conn)


def consolidate_single_experiment(prefix, expid):
    output_dir = '%s/conf/exp%d/output/' % (prefix, expid)
    csv_out_fh = open(os.path.join(output_dir, "summary%d.csv" % expid), "w")
    conf_dir = '%s/conf/exp%d/exp%d_base-variants' % (prefix, expid, expid)
    output_db_conn = get_susx_mysql_conn()
    if output_db_conn:
        writer = ConsolidatedResultsSqlAndCsvWriter(expid, csv_out_fh, output_db_conn)
    else:
        writer = ConsolidatedResultsCsvWriter(csv_out_fh)
    consolidate_results(writer, conf_dir, output_dir )


if __name__ == '__main__':
    # ----------- CONSOLIDATION -----------
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:\t%(message)s")

    prefix = '/mnt/lustre/scratch/inf/mmb28/thesisgenerator'
    # consolidate_single_experiment(prefix, 0)
    for expid in range(1, 129):
        consolidate_single_experiment(prefix, expid)
