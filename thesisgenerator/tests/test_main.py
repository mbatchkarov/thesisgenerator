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
        self.vector_source = PrecomputedSimilaritiesVectorSource(**self._thesaurus_opts)

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
            'k': 'all'
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

        self.pruned_training_matrix = np.array([
            [1, 1, 0],
            [0, 0, 1],
        ])
        self.pruned_vocab = {'a/n': 0, 'b/n': 1, 'd/n': 2}
        self.full_vocab = {'a/n': 0, 'b/n': 1, 'c/n': 2, 'd/n': 3, 'e/n': 4, 'f/n': 5}

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

    def _vectorize_data(self, vector_source=None):
        # at this point self._load_data should have been called and as a result the fields
        # self.x_tr, y_tr, x_test and y_test must have been initialised
        # also, self.tokenizer and self.thesaurus must have been initialised
        if vector_source:
            #pipeline.named_steps['vect'].thesaurus_getter = thesaurus_getter
            self.vector_source = vector_source

        pipeline = __main__._build_pipeline(
            self.vector_source,
            12345, #id for naming debug files
            None, # classifier
            self.feature_extraction_conf,
            self.feature_selection_conf,
            {'run': False}, # dim re. conf
            None, # classifier options
            '.', # temp files dir
            True, # debug mode
            'tests' # name of experiments
        )

        raw_data = (self.x_tr, self.y_tr, self.x_ev, self.y_ev)

        x_tr, y_tr, x_test, y_test = tokenize_data(raw_data, self.tokenizer, self.dataset_names)

        x1 = pipeline.fit_transform(x_tr, y_tr)

        voc = pipeline.named_steps['fs'].vocabulary_
        x2 = pipeline.transform(x_test)

        return x1, x2, voc

    def _reload_thesaurus_and_tokenizer(self):
        self.vector_source = PrecomputedSimilaritiesVectorSource(**self._thesaurus_opts)
        self.tokenizer = tokenizers.XmlTokenizer(**self.tokenizer_opts)


    def tearDown(self):
        files = glob.glob('PostVectDump_tests*')
        for f in files:
            if os.path.exists(f):
                os.remove(f)

    def _strip(self, mydict):
        #{ ('1-GRAM', ('X',)) : int} -> {'X' : int}
        for k, v in mydict.iteritems():
            self.assertEquals(len(k), 2)
        return {k[1][0]: v for k, v in mydict.iteritems()}

    def test_baseline_use_all_features_signifier_only_23(self):
        self.feature_extraction_conf['vocab_from_thes'] = False
        self._thesaurus_opts['thesaurus_files'] = \
            ['thesisgenerator/resources/exp0-0b.strings']
        self._reload_thesaurus_and_tokenizer()

        self.x_tr, self.y_tr, self.x_ev, self.y_ev = self. \
            _load_data('thesisgenerator/resources/test-baseline')

        x1, x2, voc = self._vectorize_data()

        self.assertDictEqual({'a/n': 0, 'b/n': 1, 'c/n': 2,
                              'd/n': 3, 'e/n': 4, 'f/n': 5},
                             self._strip(voc))

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

        self.assertDictEqual(self.pruned_vocab, self._strip(voc))

        #self.assertIsInstance(x1, sp.spmatrix)
        t.assert_array_equal(
            x1.toarray(),
            self.pruned_training_matrix
        )

        t.assert_array_equal(
            x2.toarray(),
            np.array(
                [# todo wtf is this, how can features be removed and yet the vocabulary stays the same
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
                             self._strip(voc))

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

        self.assertDictEqual(self.pruned_vocab, self._strip(voc))

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

        self.assertDictEqual(self.full_vocab,
                             self._strip(voc))

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

        self.assertDictEqual(self.pruned_vocab, self._strip(voc))

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

        self.assertDictEqual(self.full_vocab,
                             self._strip(voc))

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
        source.vocab = voc
        x1, x2, voc = self._vectorize_data(source)
        self.assertAlmostEqual(x2.sum(), 11.0)
        self.assertGreater(std(x2.todense()), 0)
        # seven tokens will be looked up, with random in-vocabulary neighbours
        # returned each time. Std>0 shows that it's not the same thing
        # returned each time
        print x2