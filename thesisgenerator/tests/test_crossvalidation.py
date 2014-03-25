from collections import Counter
from unittest import TestCase
from itertools import groupby
from operator import itemgetter
from numpy import array
from numpy.testing import assert_array_equal
from thesisgenerator.__main__ import _build_crossvalidation_iterator

__author__ = 'mmb28'

# todo we're missing tests for several types of crossvalidation, such as
# 'oracle', 'skfold' and 'bootstrap'. I assume scikit-learn implements
# these correctly and I am using their API correctly, too

class TestUtils(TestCase):
    def setUp(self):
        # The data set consists of two parts, 'train' and 'test'. In some cases,
        # we use 'train' for training and 'test' for testing. Sometimes,
        # we use (different) bits of 'train' for BOTH training and testing,
        # and 'test' for parameter tuning

        # 'train' section
        self.x_train = array(['a'] * 50 + ['b'] * 50 + ['c'] * 50 + ['d'] * 50)
        self.y_train = array([1] * 100 + [0] * 100)

        # 'test' section
        self.x_test = array(['e'] * 50 + ['f'] * 50 + ['g'] * 50 + ['h'] * 50)
        self.y_test = array([0] * 100 + [1] * 100)

        self.conf = {
            'type': 'test_set',
            'k': 2,
            'validation_slices': '',
            'ratio': 0.5,
            'sample_size': 10,
            'random_state': 0
        }

    def test_get_crossvalidation_iterator_with_test_set(self):
        """
        Test that the test_set CV iterator always yields a single pair of
        train-test data (which are known in advance), regardless of the seed
        """

        # with a test set the random seed should make no difference
        for seed in range(10):
            self.conf['random_state'] = seed
            it, all_x, all_y = \
                _build_crossvalidation_iterator(self.conf, self.x_train,
                                                self.y_train,
                                                x_test=self.x_test,
                                                y_test=self.y_test)

            count = 0
            for train, test in it:
                assert_array_equal(array(all_x)[train], self.x_train)
                assert_array_equal(array(all_x)[test], self.x_test)
                assert_array_equal(array(all_y)[train], self.y_train)
                assert_array_equal(array(all_y)[test], self.y_test)
                count += 1
            self.assertEqual(1, count,
                             'Test set CV iterators should only yield '
                             'one pair of (train, test) data sets')

    def test_get_crossvalidation_iterator_with_kfold(self):
        def go(subsampling=False):
            train_sets = {}
            test_sets = {}

            # should get the same result if using the same seed
            for seed in [1, 2, 3, 1, 2, 3, 1, 2, 3]:
                self.conf['random_state'] = seed
                if not subsampling:
                    it, all_x, all_y = \
                        _build_crossvalidation_iterator(self.conf,
                                                        self.x_train,
                                                        self.y_train)
                else:
                    it, all_x, all_y = \
                        _build_crossvalidation_iterator(self.conf,
                                                        self.x_train,
                                                        self.y_train,
                                                        x_test=self.x_test,
                                                        y_test=self.y_test)

                for id, (train, test) in enumerate(it):
                    tr = array(all_x)[train]
                    ev = array(all_x)[test]

                    if not subsampling:
                        # 2-fold CV, iterator should yield train/test segments
                        #  half as long as the full data set
                        self.assertEqual(len(tr),
                                         len(self.x_train) / self.conf['k'])
                    else:
                        #
                        self.assertEqual(len(tr), self.conf['sample_size'])

                    # the test bit of the corpus must not be touched
                    self.assertSetEqual(set(),
                                        set(self.x_test).intersection(set(tr)))

                    # must yield the same sets in the same order
                    if train_sets.get((seed, id)) is None:
                        # first time, just save it
                        train_sets[(seed, id)] = tr
                        test_sets[(seed, id)] = ev
                    else:
                        # after that, check for equality
                        assert_array_equal(train_sets[(seed, id)], tr)

            if not subsampling:
                # for a given seed, all fold produces must be distinct
                for seed, train_lists_keys in groupby(train_sets.keys(),
                                                      itemgetter(0)):
                    my_keys = [x for x in train_lists_keys]
                    inters = set.intersection(*[set(train_sets[x])
                                                for x in my_keys])
                    self.assertSetEqual(set(), inters)
            else:
                for x in train_sets.values():
                    # each subset used for training must contain all
                    # characters from the full set at least once
                    self.assertSetEqual(set(), set(self.x_train) - set(x))
                for x in test_sets.values():
                    # the ratios between the different classes must be
                    # approx. preserved. in our case, this means 1:1:1:1
                    c = Counter(list(x))
                    range = max(c.values()) - min(c.values())
                    self.assertLess(range, 5)

        self.conf['type'] = 'kfold'
        self.conf['k'] = 2
        go()

        self.conf['type'] = 'subsampled_test_set'
        self.conf['sample_size'] = 120
        go(subsampling=True)

    def test_seed_consistency(self):
        """
        Test that all estimators in a run receive the same data, so that their
        results are comparable
        """

        from thesisgenerator.plugins.experimental_utils import run_experiment
        from pandas import read_csv

        run_experiment(14, num_workers=1, predefined_sized=[2],
                       prefix='thesisgenerator/resources')

        df = read_csv('thesisgenerator/resources/conf/exp14/output/tests-exp14-0.out-raw.csv',
                      index_col=0)
        grouped = df.groupby(['cv_no', 'metric'])
        for name, group in grouped:
            classifiers, scores = group['classifier'], group['score']

            # only two classifier have been run
            self.assertSetEqual(set(classifiers),
                                set(['DataHashingNaiveBayes',
                                     'DataHashingLR']))

            # the hashes obtained by the two classifiers must be the same
            score1, score2 = [score for score in scores.to_dict().values()]
            self.assertEqual(score1, score2)