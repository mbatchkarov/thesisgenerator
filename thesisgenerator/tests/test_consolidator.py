# -*- coding: utf8 -*-
from itertools import chain
import logging
import os
from unittest import TestCase, skip
from glob import glob
from discoutils.thesaurus_loader import Thesaurus
from thesisgenerator.plugins.dumpers import consolidate_single_experiment

from thesisgenerator.plugins.experimental_utils import run_experiment
from thesisgenerator.utils.misc import get_susx_mysql_conn

__author__ = 'mmb28'


class TestConsolidator(TestCase):
    def setUp(cls):
        prefix = 'thesisgenerator/resources'
        # load a unigram thesaurus
        thes = Thesaurus.from_tsv('thesisgenerator/resources/exp0-0a.strings')
        print 'RES:', run_experiment(0, num_workers=1, predefined_sized=[3, 3, 3],
                                     prefix=prefix, thesaurus=thes)
        consolidate_single_experiment(prefix, 0)

    def tearDown(self):
        """
        Remove the debug files produced by this test
        """
        files = chain(glob('./PostVectDump_tests-exp0*csv'), glob('tests-*-pipeline.pickle'))
        for f in files:
            os.remove(f)

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
    def test_extract_thesausus_coverage_info(self):
        cursor = get_susx_mysql_conn().cursor()

        cursor.execute('SELECT DISTINCT classifier from data0;')
        res = cursor.fetchall()
        self.assertEqual(len(res), 2)

        # values below copied from output file before consolidation, Naive Bayes
        # is not the object under test here

        # true labels = [0 0 1]
        # predicted labels = [0 0 0]
        # this is because of the higher prior of class 0 and the fact that
        # the only feature of the test document of class 1 has only occurred
        # in class 0
        expected = (
            ('precision_score-earn', (2. / 3, 0)),
            ('precision_score-not-earn', (0, 0)),
            ('recall_score-earn', (1, 0)),
            ('recall_score-not-earn', (0, 0)),
            ('f1_score-earn', (0.8, 0)),
            ('f1_score-not-earn', (0, 0))
        )
        # all std set to -1 to indicate only a single experiment was run
        # 0 may have suggested multiple experiments with identical results
        for variable, (expected_value, exp_std) in expected:
            sql = 'select score_mean, score_std from data0 WHERE ' \
                  'metric = "{}";'.format(variable)
            cursor.execute(sql)
            res = cursor.fetchall()
            logging.info('Testing that {} == {} ± {}'.format(variable, expected_value, exp_std))
            self.assertAlmostEqual(res[0][0], expected_value, 5)
            self.assertAlmostEqual(res[0][1], exp_std, 5)

        sql = 'select distinct name from data0 ORDER BY name;'
        cursor.execute(sql)
        res = cursor.fetchall()
        # three data sizes, three output files
        for i in range(3):
            self.assertEqual(res[i][0], 'tests-exp0-{}'.format(i))

