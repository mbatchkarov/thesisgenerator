# coding=utf-8
from collections import defaultdict
import glob
import os

from unittest import TestCase, skip

import numpy as np
from numpy.ma import std
import numpy.testing as t
import scipy.sparse as sp

from thesisgenerator.plugins import tokenizers, thesaurus_loader
from thesisgenerator import __main__
from thesisgenerator.utils.misc import _vocab_neighbour_source
from thesisgenerator.utils.data_utils import load_text_data_into_memory, tokenize_data


def _get_constant_thesaurus(vocab=None):
    """
    Returns a thesaurus-like object which has a single neighbour for
    every possible entry
    """

    def constant_thesaurus():
        return [('b/n', 1)]

    return defaultdict(constant_thesaurus)


class Test_ThesaurusVectorizer(TestCase):
    def setUp(self):
        """
        Initialises the state of helper modules to sensible defaults
        """
        self._thesaurus_opts = {
            'thesaurus_files': ['thesisgenerator/resources/exp0-0a.strings'],
            'sim_threshold': 0,
            'include_self': False
        }
        self.thesaurus = thesaurus_loader.Thesaurus(**self._thesaurus_opts)

        self.tokenizer_opts = {
            'normalise_entities': False,
            'use_pos': True,
            'coarse_pos': True,
            'lemmatize': True,
            'lowercase': True,
            'keep_only_IT': False,
            'remove_stopwords': False,
            'remove_short_words': False,
            'use_cache': False,
            'thesaurus': self.thesaurus
        }
        self.tokenizer = tokenizers.XmlTokenizer(**self.tokenizer_opts)

        self.feature_extraction_conf = {
            'vectorizer': 'thesisgenerator.plugins.bov.ThesaurusVectorizer',
            'analyzer': 'ngram',
            'use_tfidf': False,
            'min_df': 1,
            'lowercase': False,
            'record_stats': True,
            'k': 10, # use all thesaurus entries
            'train_token_handler': 'thesisgenerator.plugins.bov_feature_handlers.BaseFeatureHandler',
            'decode_token_handler': 'thesisgenerator.plugins.bov_feature_handlers.BaseFeatureHandler'
        }

        self.default_prefix = 'thesisgenerator/resources/test'

        self.data_options = {
            'input': 'content',
            'shuffle_targets': False,
            'input_generator': '',
        }

        self.x_tr, self.y_tr, self.x_ev, self.y_ev = self. \
            _load_data(self.default_prefix)

        self.training_matrix = np.array([
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
        ])

    def _load_data(self, prefix):
        """
        Loads a predefined dataset from disk
        """
        tr = '%s-tr' % prefix
        self.data_options['training_data'] = tr
        ev = '%s-ev' % prefix
        self.data_options['test_data'] = ev
        self.dataset_names = (tr, ev)
        data, _ = load_text_data_into_memory(self.data_options)
        return data

    def test_get_data_iterators(self):
        """
        Tests if data loading functions returns the right number of documents
         and the right targets
        """
        x_tr, y_tr, x_ev, y_ev = self._load_data(self.default_prefix)
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

    def _vectorize_data(self, thesaurus_getter=None):
        # at this point self._load_data should have been called and as a result the fields
        # self.x_tr, y_tr, x_test and y_test must have been initialised
        # also, self.tokenizer and self.thesaurus must have been initialised
        def fully_qualified_name(o):
            return o.__module__ + "." + o.__name__

        if thesaurus_getter:
            #pipeline.named_steps['vect'].thesaurus_getter = thesaurus_getter
            self.feature_extraction_conf['decode_thesaurus'] = fully_qualified_name(thesaurus_getter)

        pipeline = __main__._build_pipeline(
            self.thesaurus,
            12345, #id for naming debug files
            None, # classifier
            self.feature_extraction_conf,
            {'run': False}, # feature selection conf
            {'run': False}, # dim re. conf
            None, # classifier options
            '.', # temp files dir
            True, # debug mode
            'tests' # name of experiments
        )

        raw_data = (self.x_tr, self.y_tr, self.x_ev, self.y_ev)
        keep_only_IT = self.tokenizer_opts['keep_only_IT']
        x_tr, y_tr, x_test, y_test = tokenize_data(raw_data, self.tokenizer, keep_only_IT, self.dataset_names)

        x1 = pipeline.fit_transform(x_tr, y_tr)

        voc = pipeline.named_steps['vect'].vocabulary_
        x2 = pipeline.transform(x_test)

        return x1, x2, voc

    def _reload_thesaurus_and_tokenizer(self):
        self.thesaurus = thesaurus_loader.Thesaurus(**self._thesaurus_opts)
        self.tokenizer_opts['thesaurus'] = self.thesaurus
        self.tokenizer = tokenizers.XmlTokenizer(**self.tokenizer_opts)


    def tearDown(self):
        files = glob.glob('PostVectDump_tests*')
        for f in files:
            if os.path.exists(f):
                os.remove(f)

    @skip('Do not use replace_all')
    def test_replaceAll_False_includeSelf_TrueFalse_use_signified(self):
        """
        Similar to self.test_baseline_use_all_features_with_signified_D in
        that the vectorizer is set to use a SignifierSignifiedFeatureHandler
        and all tokens are kept (incl. OOT ones)

        The difference is self.test_baseline_use_all_features_with_signified_D
        only inserts the nearest in-thesaurus neighbour instead of all of the
        neighbours it can find
        """
        self.feature_extraction_conf['replace_all'] = False

        for inc_self in [True, False]:
            # the expected matrices are the same with and without
            # include_self when replace_all=False
            self._thesaurus_opts['include_self'] = inc_self
            self.thesaurus = self._reload_thesaurus_and_tokenizer()

            x1, x2, voc = self._vectorize_data()

            # check vocabulary. For some reason it does not come out in the
            # order in which words are put in, but that is fine as long as it's
            #  the same order every time- I've added a sort to ensure that
            self.assertDictEqual({'cat/n': 0, 'dog/n': 1, 'game/n': 2,
                                  'kid/n': 3, 'like/v': 4, 'play/v': 5},
                                 voc)

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

            t.assert_array_equal(
                x2.todense(),
                np.matrix(
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
                filename = 'PostVectDump_tests_%s12345.csv' % stage
                with open(filename) as infile:
                    csv_file_contents = [x.strip() for x in
                                         infile.readlines()]

                # headers must be identical character for character
                self.assertEqual(expected[0], csv_file_contents[0])
                for line1, line2 in zip(expected[1:],
                                        csv_file_contents[1:]):
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

    @skip("Do not use replace_all for now")
    def test_replaceAll_True_includeSelf_False(self):
        self.feature_extraction_conf['replace_all'] = True
        self._thesaurus_opts['include_self'] = False
        self._reload_thesaurus_and_tokenizer()

        x1, x2, voc = self._vectorize_data()

        self.assertDictEqual({'cat/n': 0, 'dog/n': 1, 'fruit/n': 2,
                              'game/n': 3, 'kid/n': 4, 'like/v': 5,
                              'play/v': 6},
                             voc)

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

    @skip("include_self makes no sense with replace_all at decode time. "
          "The nearest neighbour of each unknown token is going to be itself,"
          " which is OOV and will not be inserted")
    def test_replaceAll_True_includeSelf_True(self):
        self.feature_extraction_conf['replace_all'] = True
        self._thesaurus_opts['include_self'] = True
        self._reload_thesaurus_and_tokenizer()

        x1, x2, voc = self._vectorize_data()

        self.assertDictEqual({'cat/n': 0, 'dog/n': 1, 'fruit/n': 2,
                              'game/n': 3, 'kid/n': 4, 'like/v': 5,
                              'play/v': 6},
                             voc)

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

    def test_baseline_use_all_features_signifier_only_23(self):
        self.feature_extraction_conf['vocab_from_thes'] = False
        # self.feature_extraction_conf['use_signifier_only'] = True
        self._thesaurus_opts['thesaurus_files'] = \
            ['thesisgenerator/resources/exp0-0b.strings']
        # self._thesaurus_opts['k'] = 1 # todo needs fixing
        self._reload_thesaurus_and_tokenizer()

        self.x_tr, self.y_tr, self.x_ev, self.y_ev = self. \
            _load_data('thesisgenerator/resources/test-baseline')

        x1, x2, voc = self._vectorize_data()

        self.assertDictEqual({'a/n': 0, 'b/n': 1, 'c/n': 2,
                              'd/n': 3, 'e/n': 4, 'f/n': 5},
                             voc)

        self.assertIsInstance(x1, sp.spmatrix)
        t.assert_array_equal(
            x1.toarray(),
            self.training_matrix
        )

        t.assert_array_equal(
            x2.toarray(),
            np.array(
                [
                    [4, 1, 2, 0, 0, 0],
                ]
            )
        )

    def test_baseline_ignore_nonthesaurus_features_signifier_only_22(self):
        self.tokenizer_opts['keep_only_IT'] = True
        self._thesaurus_opts['thesaurus_files'] = \
            ['thesisgenerator/resources/exp0-0b.strings']
        # self._thesaurus_opts['k'] = 1
        self._reload_thesaurus_and_tokenizer()

        self.x_tr, self.y_tr, self.x_ev, self.y_ev = self. \
            _load_data('thesisgenerator/resources/test-baseline')

        x1, x2, voc = self._vectorize_data()

        self.assertDictEqual({'a/n': 0, 'b/n': 1, 'c/n': 2,
                              'd/n': 3, 'e/n': 4, 'f/n': 5}, voc)

        self.assertIsInstance(x1, sp.spmatrix)
        t.assert_array_equal(
            x1.toarray(),
            self.training_matrix
        )

        t.assert_array_equal(
            x2.toarray(),
            np.array(
                [
                    [4, 1, 0, 0, 0, 0]
                ]
            )
        )

    def test_baseline_use_all_features_with__signifier_signified_25(self):
        self.tokenizer_opts['keep_only_IT'] = False
        self.feature_extraction_conf['decode_token_handler'] = \
            'thesisgenerator.plugins.bov_feature_handlers.SignifierSignifiedFeatureHandler'
        self.feature_extraction_conf['k'] = 1 # equivalent to max
        self._thesaurus_opts['thesaurus_files'] = \
            ['thesisgenerator/resources/exp0-0b.strings']
        self._reload_thesaurus_and_tokenizer()

        self.x_tr, self.y_tr, self.x_ev, self.y_ev = self. \
            _load_data('thesisgenerator/resources/test-baseline')

        x1, x2, voc = self._vectorize_data()

        self.assertDictEqual({'a/n': 0, 'b/n': 1, 'c/n': 2,
                              'd/n': 3, 'e/n': 4, 'f/n': 5},
                             voc)

        self.assertIsInstance(x1, sp.spmatrix)
        t.assert_array_equal(
            x1.toarray(),
            self.training_matrix
        )

        t.assert_array_almost_equal(
            x2.toarray(),
            np.array(
                [
                    [4, 1, 2, 2.1, 0, 0]
                ]
            )
        )

    def test_baseline_ignore_nonthesaurus_features_with_signifier_signified_24(
            self):
        self.tokenizer_opts['keep_only_IT'] = True
        self.feature_extraction_conf['decode_token_handler'] = \
            'thesisgenerator.plugins.bov_feature_handlers.SignifierSignifiedFeatureHandler'
        self.feature_extraction_conf['k'] = 1 # equivalent to max
        self._thesaurus_opts['thesaurus_files'] = \
            ['thesisgenerator/resources/exp0-0b.strings']
        self._reload_thesaurus_and_tokenizer()

        self.x_tr, self.y_tr, self.x_ev, self.y_ev = self. \
            _load_data('thesisgenerator/resources/test-baseline')

        x1, x2, voc = self._vectorize_data()

        self.assertDictEqual({'a/n': 0, 'b/n': 1, 'c/n': 2,
                              'd/n': 3, 'e/n': 4, 'f/n': 5}, voc)

        self.assertIsInstance(x1, sp.spmatrix)
        t.assert_array_equal(
            x1.toarray(),
            self.training_matrix
        )

        t.assert_array_almost_equal(
            x2.toarray(),
            np.array(
                [
                    [4, 1, 0, 2.1, 0, 0]
                ]
            )
        )

    def test_baseline_use_all_features_with_signified_27(self):
        self.tokenizer_opts['keep_only_IT'] = False
        self.feature_extraction_conf['decode_token_handler'] = \
            'thesisgenerator.plugins.bov_feature_handlers.SignifiedOnlyFeatureHandler'
        self.feature_extraction_conf['k'] = 1 # equivalent to max
        self._thesaurus_opts['thesaurus_files'] = \
            ['thesisgenerator/resources/exp0-0b.strings']
        self._reload_thesaurus_and_tokenizer()

        self.x_tr, self.y_tr, self.x_ev, self.y_ev = self. \
            _load_data('thesisgenerator/resources/test-baseline')

        x1, x2, voc = self._vectorize_data()

        self.assertDictEqual({'a/n': 0, 'b/n': 1, 'c/n': 2,
                              'd/n': 3, 'e/n': 4, 'f/n': 5},
                             voc)

        self.assertIsInstance(x1, sp.spmatrix)
        t.assert_array_equal(
            x1.toarray(),
            self.training_matrix
        )

        t.assert_array_almost_equal(
            x2.toarray(),
            np.array(
                [
                    [0, 0, 0, 4.4, 0, 0],
                ]
            )
        )

    def test_baseline_ignore_nonthesaurus_features_with_signified_26(self):
        self.tokenizer_opts['keep_only_IT'] = True
        self.feature_extraction_conf['decode_token_handler'] = \
            'thesisgenerator.plugins.bov_feature_handlers.SignifiedOnlyFeatureHandler'
        self.feature_extraction_conf['k'] = 1 # equivalent to max
        self._thesaurus_opts['thesaurus_files'] = \
            ['thesisgenerator/resources/exp0-0b.strings']
        self._reload_thesaurus_and_tokenizer()

        self.x_tr, self.y_tr, self.x_ev, self.y_ev = self. \
            _load_data('thesisgenerator/resources/test-baseline')

        x1, x2, voc = self._vectorize_data()

        self.assertDictEqual({'a/n': 0, 'b/n': 1, 'c/n': 2,
                              'd/n': 3, 'e/n': 4, 'f/n': 5}, voc)

        self.assertIsInstance(x1, sp.spmatrix)
        t.assert_array_equal(
            x1.toarray(),
            self.training_matrix
        )

        t.assert_array_almost_equal(
            x2.toarray(),
            np.array(
                [
                    [0, 0, 0, 4.4, 0, 0]
                ]
            )
        )

    def test_baseline_use_all_features_with_signified_random_28(self):
        self.tokenizer_opts['keep_only_IT'] = False
        self.feature_extraction_conf['decode_token_handler'] = \
            'thesisgenerator.plugins.bov_feature_handlers.SignifierRandomBaselineFeatureHandler'
        self.feature_extraction_conf['k'] = 1    # equivalent to max
        self.feature_extraction_conf['neighbour_source'] = \
            'thesisgenerator.tests.test_main._get_constant_thesaurus'
        self._reload_thesaurus_and_tokenizer()
        self.x_tr, self.y_tr, self.x_ev, self.y_ev = self. \
            _load_data('thesisgenerator/resources/test-baseline')

        x1, x2, voc = self._vectorize_data(_get_constant_thesaurus)

        self.assertDictEqual({'a/n': 0, 'b/n': 1, 'c/n': 2,
                              'd/n': 3, 'e/n': 4, 'f/n': 5},
                             voc)

        self.assertIsInstance(x1, sp.spmatrix)
        t.assert_array_equal(
            x1.toarray(),
            self.training_matrix
        )

        t.assert_array_almost_equal(
            x2.toarray(),
            np.array(
                [
                    [0, 11.0, 0, 0, 0, 0],
                ]
            )
        )
        # the thesaurus will always say the neighbour for something is
        #  b/n with a similarity of 1, and we look up 11 tokens overall in
        #  the test document


        self.feature_extraction_conf['neighbour_source'] = \
            'thesisgenerator.tests.test_main._vocab_neighbour_source'

        self.x_tr, self.y_tr, self.x_ev, self.y_ev = self. \
            _load_data('thesisgenerator/resources/test-baseline')

        x1, x2, voc = self._vectorize_data(_vocab_neighbour_source)
        self.assertAlmostEqual(x2.sum(), 11.0)
        self.assertGreater(std(x2.todense()), 0)
        # seven tokens will be looked up, with random in-vocabulary neighbours
        # returned each time. Std>0 shows that it's not the same thing
        # returned each time
        print x2