# coding=utf-8

from unittest import TestCase

import numpy as np
import numpy.testing as t
import scipy.sparse as sp

from thesisgenerator.plugins import tokenizers, thesaurus_loader
from thesisgenerator.__main__ import _get_data_iterators
from thesisgenerator.__main__ import _build_pipeline


class Test_ThesaurusVectorizer(TestCase):
    def setUp(self):
        """
        Initialises the state of helper modules to sensible defaults
        """
        tokenizers.normalise_entities = False
        tokenizers.use_pos = True
        tokenizers.coarse_pos = True
        tokenizers.lemmatize = True
        tokenizers.lowercase = True

        thesaurus_loader.thesaurus_files = \
            ['sample-data/simple.thesaurus.strings']
        thesaurus_loader.sim_threshold = 0
        thesaurus_loader.k = 10
        thesaurus_loader.include_self = False
        thesaurus_loader.use_cache = False

        self.feature_extraction_conf = {
            'vectorizer': 'plugins.bov.ThesaurusVectorizer',
            'use_tfidf': False,
            'min_df': 1,
            'lowercase': False,
            'replace_all': False
        }
        self.x_tr, self.y_tr, self.x_ev, self.y_ev = self._load_data()

    def _load_data(self):
        """
        Loads a predefined dataset from disk
        """
        options = {
            'input': 'content',
            'shuffle_targets': False,
            'input_generator': '',
            'source': 'sample-data/test-tr'
        }
        x_tr, y_tr = _get_data_iterators(**options)
        options['source'] = 'sample-data/test-ev'
        x_ev, y_ev = _get_data_iterators(**options)
        return x_tr, y_tr, x_ev, y_ev

    def test_get_data_iterators(self):
        """
        Tests if data loading functions returns the right number of documents
         and the right targets
        """
        x_tr, y_tr, x_ev, y_ev = self._load_data()
        # both training and testing data contain three docs
        self.assertEqual(len(x_tr), 3)
        self.assertEqual(len(x_ev), 3)
        self.assertEqual(len(y_ev), 3)
        self.assertEqual(len(y_tr), 3)

        # one doc of class 1 and two of class 0
        for y in [y_tr, y_ev]:
            self.assertEqual(y[0], 0)
            self.assertEqual(y[1], 0)
            self.assertEqual(y[2], 1)

    def _build_simple_pipeline(self, feature_extraction_conf):
        """
        Builds a sklearn pipeline consisting of a single
        """
        # todo this does not pass the values of feature_extraction_conf in,
        # but instead creates a pipe using the no-params constructor and then
        #  assigns the parameters to a copy

        pipeline = _build_pipeline(12345, #id, for naming debug files
                                   None, # classifier
                                   feature_extraction_conf,
                                   {'run': False}, # feature selection conf
                                   {'run': False}, # dim re. conf
                                   None, # classifier options
                                   '.', # temp files dir
                                   True  # debug mode
        )
        return pipeline

    def test_includeSelf_TrueFalse(self):
        self.feature_extraction_conf['replace_all'] = False

        for inc_self in [True, False]:
            # the expected matrices are the same with and without
            # include_self when replace_all=False
            thesaurus_loader.include_self = inc_self

            pipeline = self._build_simple_pipeline(self.feature_extraction_conf)
            self.assertEqual(len(pipeline.named_steps), 2)
            self.assertEqual(sorted(list(pipeline.named_steps))[0], 'dumper')
            self.assertEqual(sorted(list(pipeline.named_steps))[1], 'vect')

            x1 = pipeline.fit_transform(self.x_tr, self.y_tr)

            # check vocabulary. For some reason it does not come out in the order
            #  in which words are put in, but that is fine as long as its the
            # same order every time
            self.assertDictEqual({'cat/n': 0, 'dog/n': 1, 'game/n': 2,
                                  'kid/n': 3, 'like/v': 4, 'play/v': 5},
                                 pipeline.named_steps['vect'].vocabulary_)

            # test output when not replacing all feature (old model)
            self.assertIsInstance(x1, sp.spmatrix)
            t.assert_array_equal(
                x1.todense(),
                np.array(
                    [
                        [1, 1, 0, 0, 1, 0],
                        [1, 1, 0, 0, 1, 0],
                        [0, 0, 1, 1, 0, 1]
                    ]
                )
            )

            x2 = pipeline.transform(self.x_ev)
            t.assert_array_equal(
                x2.todense(),
                np.array(
                    [
                        [.06, .05, 0, 0, 1, 0],
                        [.06, .05, 0, 0, 1, 0],
                        [0, 1, 0, 0, 0, 0]
                    ]
                )
            )


            # ===============================================================
            # Test that the CSV dumper correctly unpacks the vocabulary and
            # the feature vectors
            # ===============================================================
            def compare_csv(expected, stage):
                expected = [x.strip() for x in expected.split()]
                with open('PostVectDump-%s12345.csv' % stage) as infile:
                    csv_file_contents = [x.strip() for x in infile.readlines()]

                # headers must be identical character for character
                self.assertEqual(expected[0], csv_file_contents[0])
                for line1, line2 in zip(expected[1:], csv_file_contents[1:]):
                    for token1, token2 in zip(line1.split(','),
                                              line2.split(',')):
                        try:
                            print token1, token2
                            self.assertEqual(float(token1), float(token2))
                        except ValueError:
                            # in the evaluation file there are no targets,
                            # i.e. token1==''
                            self.assertEqual(token1, token2)
                            self.assertEqual(token1.strip(), '')

            expected = \
                """
    id,target,total_feat_weight,nonzero_feats,cat/n,dog/n,game/n,kid/n,like/v,play/v
    0,0,3,3,1,1,0,0,1,0
    1,0,3,3,1,1,0,0,1,0
    2,1,3,3,0,0,1,1,0,1
                """
            compare_csv(expected, 'tr')

            expected = \
                """
    id,target,total_feat_weight,nonzero_feats,cat/n,dog/n,game/n,kid/n,like/v,play/v
    0,,1.11,3,0.06,0.05,0,0,1,0
    1,,1.11,3,0.06,0.05,0,0,1,0
    2,,1,1,0,1,0,0,0,0
                """
            compare_csv(expected, 'ev')

    def test_replaceAll_True_includeSelf_False(self):
        self.feature_extraction_conf['replace_all'] = True
        thesaurus_loader.include_self = False

        pipeline = self._build_simple_pipeline(self.feature_extraction_conf)
        x1 = pipeline.fit_transform(self.x_tr, self.y_tr)

        self.assertDictEqual({'cat/n': 0, 'dog/n': 1, 'fruit/n': 2,
                              'game/n': 3, 'kid/n': 4, 'like/v': 5,
                              'play/v': 6},
                             pipeline.named_steps['vect'].vocabulary_)

        self.assertIsInstance(x1, sp.spmatrix)
        t.assert_array_equal(
            x1.toarray(),
            np.array(
                [
                    [.7, .8, 0, .4, 0.3, 0, 0.11],
                    [.7, .8, 0, .4, 0.3, 0, 0.11],
                    [.06, .04, 0.1, 0, 0, 0.09, 0.6]
                ]
            )
        )

        x2 = pipeline.transform(self.x_ev)
        t.assert_array_equal(
            x2.toarray(),
            np.array(
                [
                    [.06, .05, 0, 0, 0, 0, 0.11],
                    [.06, .05, 0, 0, 0, 0, 0.11],
                    [.7, 0, 0, 0, 0.3, 0, 0]
                ]
            )
        )

    def test_replaceAll_True_includeSelf_True(self):
        self.feature_extraction_conf['replace_all'] = True
        thesaurus_loader.include_self = True

        pipeline = self._build_simple_pipeline(self.feature_extraction_conf)
        x1 = pipeline.fit_transform(self.x_tr, self.y_tr)

        self.assertDictEqual({'cat/n': 0, 'dog/n': 1, 'fruit/n': 2,
                              'game/n': 3, 'kid/n': 4, 'like/v': 5,
                              'play/v': 6},
                             pipeline.named_steps['vect'].vocabulary_)

        self.assertIsInstance(x1, sp.spmatrix)
        t.assert_array_equal(
            x1.toarray(),
            np.array(
                [
                    [1.7, 1.8, 0, .4, 0.3, 1, 0.11],
                    [1.7, 1.8, 0, .4, 0.3, 1, 0.11],
                    [.06, .04, 0.1, 1, 1, 0.09, 1.6]
                ]
            )
        )

        x2 = pipeline.transform(self.x_ev)
        t.assert_array_equal(
            x2.toarray(),
            np.array(
                [
                    [.06, .05, 1, 0, 0, 1, 0.11],
                    [.06, .05, 1, 0, 0, 1, 0.11],
                    [.7, 1, 0, 0, 0.3, 0, 0]
                ]
            )
        )

