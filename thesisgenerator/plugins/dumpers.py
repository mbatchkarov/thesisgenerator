# coding=utf-8
from collections import defaultdict
import csv
import logging
import os
from datetime import datetime as dt
from numpy import count_nonzero
import sys
from glob import glob

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
from sklearn.base import TransformerMixin
import pandas as pd
from discoutils.cmd_utils import get_git_hash
from thesisgenerator.utils.misc import get_susx_mysql_conn
from thesisgenerator.utils.conf_file_utils import parse_config_file
from thesisgenerator.utils import db

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

           # performance
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

    def consolidate_results(self, conf_dir, output_dir):
        """
        Consolidates the results of a series of experiment and passes it on to a
        writer
        A single thesaurus must be used in each experiment
        """
        print('Consolidating results from %s' % conf_dir)

        conf_file = glob(os.path.join(conf_dir, '*.conf'))[0]
        print('Processing file %s' % conf_file)

        config_obj, configspec_file = parse_config_file(conf_file)

        exp_name = config_obj['name']
        cv_folds = config_obj['crossvalidation']['k']
        sample_size = config_obj['crossvalidation']['sample_size']

        # find out the classifier score from the final csv file
        output_file = os.path.join(output_dir, '%s.out.csv' % exp_name)
        git_hash = get_git_hash()

        try:
            reader = csv.reader(open(output_file, 'r'))
            _ = next(reader)  # skip over header
            for row in reader:
                classifier, metric, score_my_mean, score_my_std = row

                self.c.writerow([
                    None,  # primary key, should be updated automatically
                    exp_name,
                    git_hash,
                    dt.now().isoformat(),

                    # experiment settings
                    cv_folds,
                    sample_size,  # sample_size
                    classifier,
                    # these need to be converted to a bool and then to an int
                    # because mysql stores booleans as a tinyint and complains
                    # if you pass in a python boolean

                    # performance
                    metric,
                    score_my_mean,
                    score_my_std])
        except IOError:
            print('WARNING: output file %s is missing' % output_file)


def consolidate_single_experiment(prefix, expid):
    output_dir = '%s/conf/exp%d/output/' % (prefix, expid)
    csv_out_fh = open(os.path.join(output_dir, "summary%d.csv" % expid), "w")
    conf_dir = '%s/conf/exp%d/exp%d_base-variants' % (prefix, expid, expid)
    writer = ConsolidatedResultsCsvWriter(csv_out_fh)
    writer.consolidate_results(conf_dir, output_dir)

    # insert a subset of the data to MySQL
    if expid == 0:
        # can't have experiment ID 0 in mysql, so can't have the results point to it
        return

    output_db_conn = get_susx_mysql_conn()
    if output_db_conn:
        # do some SQL-fu here
        output_file = os.path.join(output_dir, 'exp%d-0.out.csv' % expid)
        df = pd.read_csv(output_file)
        data = {'id': expid}
        for classifier in set(df.classifier):
            data['classifier'] = classifier
            data['accuracy_mean'] = df['score_mean'][(df['classifier'] == classifier) &
                                                     (df['metric'] == 'accuracy_score')].iloc[0]
            data['accuracy_std'] = df['score_std'][(df['classifier'] == classifier) &
                                                   (df['metric'] == 'accuracy_score')].iloc[0]
            data['microf1_mean'] = df['score_mean'][(df['classifier'] == classifier) &
                                                    (df['metric'] == 'microavg_f1')].iloc[0]
            data['microf1_std'] = df['score_std'][(df['classifier'] == classifier) &
                                                  (df['metric'] == 'microavg_f1')].iloc[0]
            data['macrof1_mean'] = df['score_mean'][(df['classifier'] == classifier) &
                                                    (df['metric'] == 'macroavg_f1')].iloc[0]
            data['macrof1_std'] = df['score_std'][(df['classifier'] == classifier) &
                                                  (df['metric'] == 'macroavg_f1')].iloc[0]
        res = db.Results(**data)
        res.save()


if __name__ == '__main__':
    # ----------- CONSOLIDATION -----------
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:\t%(message)s")

    prefix = '/mnt/lustre/scratch/inf/mmb28/thesisgenerator'
    # consolidate_single_experiment(prefix, 0)
    for expid in [0]:  # range(1, 129):
        consolidate_single_experiment(prefix, expid)
