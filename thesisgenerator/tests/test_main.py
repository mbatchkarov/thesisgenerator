# coding=utf-8
import glob
import os

from unittest import TestCase

import numpy as np
from numpy.ma import std
import numpy.testing as t
import scipy.sparse as sp
from thesisgenerator.composers.vectorstore import PrecomputedSimilaritiesVectorSource, ConstantNeighbourVectorSource

from thesisgenerator.plugins import tokenizers
from thesisgenerator import __main__
from thesisgenerator.tests.test_feature_selectors import strip
from thesisgenerator.utils.data_utils import load_text_data_into_memory, tokenize_data


class TestThesaurusVectorizer(TestCase):
    def setUp(self):
        """
        Initialises the state of helper modules to sensible defaults
        """
        self._thesaurus_opts = {
            'thesaurus_files': ['thesisgenerator/resources/exp0-0a.strings'],
            'sim_threshold': 0,
            'include_self': False
        }
        self.vector_source = PrecomputedSimilaritiesVectorSource.from_file(**self._thesaurus_opts)

        self.tokenizer_opts = {
            'normalise_entities': False,
            'use_pos': True,
            'coarse_pos': True,
            'lemmatize': True,
            'lowercase': True,
            'remove_stopwords': False,
            'remove_short_words': False,
            'use_cache': False
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
        self.feature_selection_conf = {
            'run': True,
            'method': 'thesisgenerator.composers.feature_selectors.VectorBackedSelectKBest',
            'scoring_function': 'sklearn.feature_selection.chi2',
            'ensure_vectors_exist': False,
            'k': 'all',
            'vector_source': None
        }

        self.default_prefix = 'thesisgenerator/resources/test'

        self.x_tr, self.y_tr, self.x_ev, self.y_ev = self. \
            _load_data(self.default_prefix)

        self.training_matrix = np.array([
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
        ])

        self.pruned_training_matrix = np.array([
            [1, 1, 0],
            [0, 0, 1],
        ])
        self.pruned_vocab = {'a/N': 0, 'b/N': 1, 'd/N': 2}
        self.full_vocab = {'a/N': 0, 'b/N': 1, 'c/N': 2, 'd/N': 3, 'e/N': 4, 'f/N': 5}

    def _load_data(self, prefix):
        """
        Loads a predefined dataset from disk
        """
        tr = '%s-tr' % prefix
        ev = '%s-ev' % prefix
        self.dataset_names = (tr, ev)
        data, _ = load_text_data_into_memory(tr, ev)
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

    def _vectorize_data(self, vector_source=None):
        # at this point self._load_data should have been called and as a result the fields
        # self.x_tr, y_tr, x_test and y_test must have been initialised
        # also, self.tokenizer and self.thesaurus must have been initialised
        if vector_source:
            self.vector_source = vector_source

        pipeline, fit_params = __main__._build_pipeline(
            12345, #id for naming debug files
            self.vector_source,
            # None, # classifier
            self.feature_extraction_conf,
            self.feature_selection_conf,
            {'run': False}, # dim re. conf
            # None, # classifier options
            '.', # temp files dir
            True, # debug mode
            'test_main' # name of experiments
        )

        raw_data = (self.x_tr, self.y_tr, self.x_ev, self.y_ev)

        x_tr, y_tr, x_test, y_test = tokenize_data(raw_data, self.tokenizer, self.dataset_names)

        x1 = pipeline.fit_transform(x_tr, y_tr, **fit_params)

        voc = pipeline.named_steps['fs'].vocabulary_
        x2 = pipeline.transform(x_test)

        return x1, x2, voc

    def _reload_thesaurus_and_tokenizer(self):
        self.vector_source = PrecomputedSimilaritiesVectorSource.from_file(**self._thesaurus_opts)
        self.feature_selection_conf['vector_source'] = self.vector_source
        self.tokenizer = tokenizers.XmlTokenizer(**self.tokenizer_opts)


    def tearDown(self):
        files = glob.glob('PostVectDump_tests*')
        for f in files:
            if os.path.exists(f):
                os.remove(f)

    def test_baseline_use_all_features_signifier_only_23(self):
        self.feature_extraction_conf['vocab_from_thes'] = False
        self._thesaurus_opts['thesaurus_files'] = \
            ['thesisgenerator/resources/exp0-0b.strings']
        self._reload_thesaurus_and_tokenizer()

        self.x_tr, self.y_tr, self.x_ev, self.y_ev = self. \
            _load_data('thesisgenerator/resources/test-baseline')

        x1, x2, voc = self._vectorize_data()
        self.assertDictEqual(self.full_vocab, strip(voc))

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
        self.feature_selection_conf['ensure_vectors_exist'] = True
        self._thesaurus_opts['thesaurus_files'] = ['thesisgenerator/resources/exp0-0b.strings']
        self._reload_thesaurus_and_tokenizer()

        self.x_tr, self.y_tr, self.x_ev, self.y_ev = self._load_data('thesisgenerator/resources/test-baseline')

        x1, x2, voc = self._vectorize_data()

        self.assertDictEqual(self.pruned_vocab, strip(voc))

        #self.assertIsInstance(x1, sp.spmatrix)
        t.assert_array_equal(
            x1.toarray(),
            self.pruned_training_matrix
        )

        t.assert_array_equal(
            x2.toarray(),
            np.array(
                [
                    [4, 1, 0]
                ]
            )
        )

    def test_baseline_use_all_features_with__signifier_signified_25(self):
        self.feature_selection_conf['ensure_vectors_exist'] = False
        self.feature_extraction_conf['decode_token_handler'] = \
            'thesisgenerator.plugins.bov_feature_handlers.SignifierSignifiedFeatureHandler'
        self.feature_extraction_conf['k'] = 1 # equivalent to max
        self._thesaurus_opts['thesaurus_files'] = \
            ['thesisgenerator/resources/exp0-0b.strings']
        self._reload_thesaurus_and_tokenizer()

        self.x_tr, self.y_tr, self.x_ev, self.y_ev = self. \
            _load_data('thesisgenerator/resources/test-baseline')

        x1, x2, voc = self._vectorize_data()

        self.assertDictEqual(self.full_vocab,
                             strip(voc))

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
        self.feature_selection_conf['ensure_vectors_exist'] = True
        self.feature_extraction_conf['decode_token_handler'] = \
            'thesisgenerator.plugins.bov_feature_handlers.SignifierSignifiedFeatureHandler'
        self.feature_extraction_conf['k'] = 1 # equivalent to max
        self._thesaurus_opts['thesaurus_files'] = \
            ['thesisgenerator/resources/exp0-0b.strings']
        self._reload_thesaurus_and_tokenizer()

        self.x_tr, self.y_tr, self.x_ev, self.y_ev = self. \
            _load_data('thesisgenerator/resources/test-baseline')

        x1, x2, voc = self._vectorize_data()

        self.assertDictEqual(self.pruned_vocab, strip(voc))

        self.assertIsInstance(x1, sp.spmatrix)
        t.assert_array_equal(
            x1.toarray(),
            self.pruned_training_matrix
        )

        t.assert_array_almost_equal(
            x2.toarray(),
            np.array(
                [
                    [4, 1, 2.1]
                ]
            )
        )

    def test_baseline_use_all_features_with_signified_27(self):
        self.feature_selection_conf['ensure_vectors_exist'] = False
        self.feature_extraction_conf['decode_token_handler'] = \
            'thesisgenerator.plugins.bov_feature_handlers.SignifiedOnlyFeatureHandler'
        self.feature_extraction_conf['k'] = 1 # equivalent to max
        self._thesaurus_opts['thesaurus_files'] = \
            ['thesisgenerator/resources/exp0-0b.strings']
        self._reload_thesaurus_and_tokenizer()

        self.x_tr, self.y_tr, self.x_ev, self.y_ev = self. \
            _load_data('thesisgenerator/resources/test-baseline')

        x1, x2, voc = self._vectorize_data()

        self.assertDictEqual(self.full_vocab, strip(voc))

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
        self.feature_selection_conf['ensure_vectors_exist'] = True
        self.feature_extraction_conf['decode_token_handler'] = \
            'thesisgenerator.plugins.bov_feature_handlers.SignifiedOnlyFeatureHandler'
        self.feature_extraction_conf['k'] = 1 # equivalent to max
        self._thesaurus_opts['thesaurus_files'] = \
            ['thesisgenerator/resources/exp0-0b.strings']
        self._reload_thesaurus_and_tokenizer()

        self.x_tr, self.y_tr, self.x_ev, self.y_ev = self. \
            _load_data('thesisgenerator/resources/test-baseline')

        x1, x2, voc = self._vectorize_data()

        self.assertDictEqual(self.pruned_vocab, strip(voc))

        self.assertIsInstance(x1, sp.spmatrix)
        t.assert_array_equal(
            x1.toarray(),
            self.pruned_training_matrix
        )

        t.assert_array_almost_equal(
            x2.toarray(),
            np.array(
                [
                    [0, 0, 4.4]
                ]
            )
        )

    def test_baseline_use_all_features_with_signified_random_28(self):
        self.feature_selection_conf['ensure_vectors_exist'] = False
        self.feature_extraction_conf['decode_token_handler'] = \
            'thesisgenerator.plugins.bov_feature_handlers.SignifierRandomBaselineFeatureHandler'
        self.feature_extraction_conf['k'] = 1    # equivalent to max
        self.feature_extraction_conf['neighbour_source'] = \
            'thesisgenerator.tests.test_main._get_constant_thesaurus'
        self._reload_thesaurus_and_tokenizer()
        self.x_tr, self.y_tr, self.x_ev, self.y_ev = self. \
            _load_data('thesisgenerator/resources/test-baseline')

        source = ConstantNeighbourVectorSource()
        x1, x2, voc = self._vectorize_data(source)

        self.assertDictEqual(self.full_vocab, strip(voc))

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
        #  b/N with a similarity of 1, and we look up 11 tokens overall in
        #  the test document
        source.vocab = voc
        x1, x2, voc = self._vectorize_data(source)
        self.assertAlmostEqual(x2.sum(), 11.0)
        self.assertGreater(std(x2.todense()), 0)
        # seven tokens will be looked up, with random in-vocabulary neighbours
        # returned each time. Std>0 shows that it's not the same thing
        # returned each time
        #print x2