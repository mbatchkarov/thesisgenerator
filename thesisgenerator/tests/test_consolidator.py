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

"""
This is now (Dec 2014) a smoke test.

It used to test if information is correctly extracted from log files but:
    - I'm no longer interested in the information
    - Storage format has been changed so the test can't possibly pass
"""


class TestConsolidator(TestCase):
    def setUp(cls):
        prefix = 'thesisgenerator/resources'
        # load a unigram thesaurus
        thes = Thesaurus.from_tsv('thesisgenerator/resources/exp0-0a.strings')
        print('RES:', run_experiment(0, num_workers=1, predefined_sized=[3, 3, 3],
                                     prefix=prefix, thesaurus=thes))
        consolidate_single_experiment(prefix, 0)

    def tearDown(self):
        """
        Remove the debug files produced by this test
        """
        files = chain(glob('./PostVectDump_exp0*csv'), glob('tests-*-pipeline.pickle'))
        for f in files:
            os.remove(f)

    def test_extract_thesausus_coverage_info(self):
        self.assertEqual(0, 0)