# -*- coding: utf8 -*-
from itertools import chain
import logging
import os
from unittest import TestCase, skip
from glob import glob
from thesisgenerator.composers.vectorstore import PrecomputedSimilaritiesVectorSource
from thesisgenerator.plugins.dumpers import consolidate_single_experiment

from thesisgenerator.plugins.experimental_utils import run_experiment
from thesisgenerator.utils.misc import get_susx_mysql_conn

__author__ = 'mmb28'


class TestConsolidator(TestCase):
    def setUp(cls):
        prefix = 'thesisgenerator/resources'
        # load a mock unigram thesaurus, bypassing the similarity calculation provided by CompositeVectorSource
        vector_source = PrecomputedSimilaritiesVectorSource.from_file(
            thesaurus_files=['thesisgenerator/resources/exp0-0a.strings'])
        print 'RES:', run_experiment(0, num_workers=1, predefined_sized=[3, 3, 3],
                                     prefix=prefix, vector_source=vector_source)
        consolidate_single_experiment(prefix, 0)

    def tearDown(self):
        """
        Remove the debug files produced by this test
        """
        files = chain(glob('./PostVectDump_tests-exp0*csv'), glob('tests-*-pipeline.pickle'))
        for f in files:
            os.remove(f)

    # @skip("""
    # The current concurrency model is:
    #
    # for each training data size (now run serially, previously in parallel):
    #      ___________________________________________
    #     |                                           |
    #     |for each classifier:                       |
    #     |    for each CV fold (run in parallel):    |
    #     |       train                               |
    #     |       evaluate                            |
    #     |_________________shared log file___________|
    #
    # Information about the IV/IT token/type counts of each fold is intertwined in the log files and cannot be extracted
    # reliably. Previously, that used to be possible because all log-producing stages that share a log file were run
    # serially.
    # """)
    def test_extract_thesausus_coverage_info(self):
        cursor = get_susx_mysql_conn().cursor()

        cursor.execute('SELECT DISTINCT classifier from data0;')
        res = cursor.fetchall()
        self.assertEqual(len(res), 2)
        #self.assertEqual(res[0][0], 'MultinomialNB')

        expected = {
            'corpus': '0',
            'features': '0',
            'pos': 'a',
            'fef': '?',
            'use_tfidf': 0,
            'ensure_vectors_exist': 0,
            'train_token_handler': 'BaseFeatureHandler',
            # changing this to SignifierSignifiedFeatureHandler will not affect
            #  the vector of the third test document, i.e. will not change
            # performance
            'decode_token_handler': 'SignifierSignifiedFeatureHandler'
        }
        for variable, expected_value in expected.items():
            cursor.execute('SELECT DISTINCT %s from data0;' % variable)
            res = cursor.fetchall()
            print('Testing that {} == {}'.format(variable, expected_value))
            self.assertEqual(res[0][0], expected_value)

        # values below copied from output file before consolidation, Naive Bayes
        # is not the ubject under test here

        # true labels = [0 0 1]
        # predicted labels = [0 0 0]
        # this is because of the higher prior of class 0 and the fact that
        # the only feature of the test document of class 1 has only occurred
        # in class 0
        expected = (
            ('precision_score-class0', (2. / 3, 0)),
            ('precision_score-class1', (0, 0)),
            ('recall_score-class0', (1, 0)),
            ('recall_score-class1', (0, 0)),
            ('f1_score-class0', (0.8, 0)),
            ('f1_score-class1', (0, 0))
        )
        # all std set to -1 to indicate only a single experiment was run
        # 0 may have suggested multiple experiments with identical results
        for variable, (expected_value, exp_std) in expected:
            sql = 'select score_mean, score_std from data0 WHERE ' \
                  'metric = "{}";'.format(variable)
            cursor.execute(sql)
            res = cursor.fetchall()
            logging.info('Testing that {} == {} ± {}'.format(variable,
                                                             expected_value,
                                                             exp_std))
            self.assertAlmostEqual(res[0][0], expected_value, 5)
            self.assertAlmostEqual(res[0][1], exp_std, 5)

        sql = 'select distinct name from data0 ORDER BY name;'
        cursor.execute(sql)
        res = cursor.fetchall()
        # three data sizes, three output files
        for i in range(3):
            self.assertEqual(res[i][0], 'tests-exp0-{}'.format(i))

