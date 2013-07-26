# -*- coding: utf8 -*-

import logging
from unittest import TestCase
from thesisgenerator.plugins.experimental_utils import run_experiment
from thesisgenerator.utils import get_susx_mysql_conn

__author__ = 'mmb28'


class TestConsolidator(TestCase):
    @classmethod
    def setUpClass(cls):
        prefix = 'thesisgenerator/resources'
        run_experiment(0, num_workers=1, predefined_sized=[3],
                       prefix=prefix)

    def test_extract_thesausus_coverage_info(self):
        with open('thesisgenerator/resources/conf/exp0/output/summary0.csv') \
            as infile:
            log_txt = ''.join(infile.readlines())
        cursor = get_susx_mysql_conn().cursor()

        cursor.execute('SELECT DISTINCT classifier from data00;')
        res = cursor.fetchall()
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0][0], 'MultinomialNB')

        exp = {
            'total_tok': 9, 'total_typ': 5,
            'iv_it_tok_mean': 3, 'iv_it_tok_std': 0,
            'iv_oot_tok_mean': 0, 'iv_oot_tok_std': 0,
            'oov_it_tok_mean': 2, 'oov_it_tok_std': 0,
            'oov_oot_tok_mean': 4, 'oov_oot_tok_std': 0,
            'iv_it_typ_mean': 2, 'iv_it_typ_std': 0,
            'iv_oot_typ_mean': 0, 'iv_oot_typ_std': 0,
            'oov_it_typ_mean': 1, 'oov_it_typ_std': 0,
            'oov_oot_typ_mean': 2, 'oov_oot_typ_std': 0,
            'classifier': 'MultinomialNB',
            'corpus': '0',
            'features': '0',
            'pos': 'a',
            'fef': '?',
            'use_tfidf': 0,
            'keep_only_IT': 0,
            'train_token_handler': 'BaseFeatureHandler',
            # changing this to SignifierSignifiedFeatureHandler will not affect
            #  the vector of the third test document, i.e. will not change
            # performance
            'train_token_handler': 'BaseFeatureHandler'
        }
        for variable, exp_mean in exp.items():
            cursor.execute('SELECT DISTINCT %s from data00;' % variable)
            res = cursor.fetchall()
            logging.info('Testing that {} == {}'.format(variable, res[0][0]))
            self.assertEqual(res[0][0], exp_mean)

        # values below copied from output file before consolidation, Naive Bayes
        # is not the ubject under test here

        # true labels = [0 0 1]
        # predicted labels = [0 0 0]
        # this is because of the higher prior of class 0 and the fact that
        # the only feature of the test document of class 1 has only occurred
        # in class 0
        exp = (
            ('precision_score-class0', (2. / 3, 0)),
            ('precision_score-class1', (0, 0)),
            ('recall_score-class0', (1, 0)),
            ('recall_score-class1', (0, 0)),
            ('f1_score-class0', (0.8, 0)),
            ('f1_score-class1', (0, 0))
        )
        # all std set to -1 to indicate only a single experiment was run
        # 0 may have suggested multiple experiments with identical results
        for variable, (exp_mean, exp_std) in exp:
            sql = 'select score_mean, score_std from data00 WHERE ' \
                  'metric = "{}";'.format(variable)
            cursor.execute(sql)
            res = cursor.fetchall()
            logging.info('Testing that {} == {} ± {}'.format(variable,
                                                             exp_mean,
                                                             exp_std))
            self.assertAlmostEqual(res[0][0], exp_mean, 5)
            self.assertAlmostEqual(res[0][1], exp_std, 5)
            logging.info('OK')



