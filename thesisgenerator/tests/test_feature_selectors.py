import os
from unittest import TestCase

from sklearn.pipeline import Pipeline
import numpy as np
import numpy.testing as t
from pandas.io.parsers import read_csv

from thesisgenerator.composers.feature_selectors import VectorBackedSelectKBest
from thesisgenerator.composers.vectorstore import CompositeVectorSource, UnigramVectorSource, \
    UnigramDummyComposer, PrecomputedSimilaritiesVectorSource
from thesisgenerator.plugins.bov import ThesaurusVectorizer
from thesisgenerator.plugins.dumpers import FeatureVectorsCsvDumper
from thesisgenerator.utils.data_utils import load_text_data_into_memory, _load_tokenizer, tokenize_data


__author__ = 'mmb28'


class TestVectorBackedSelectKBest(TestCase):
    def _strip(self, mydict):
        """{ ('1-GRAM', ('X',)) : int} -> {'X' : int}"""
        for k, v in mydict.iteritems():
            self.assertEquals(len(k), 2)
        return {' '.join(k[1]): v for k, v in mydict.iteritems()}

    def setUp(self):
        self.training_matrix_signifier_bigrams = np.array(
            [[1., 1., 0., 0., 1., 0., 1., 0., 1., 0.],
             [1., 1., 0., 0., 1., 0., 1., 0., 1., 0.],
             [0., 0., 1., 1., 0., 1., 0., 1., 0., 1.]])

    def _do_feature_selection(self, ensure_vectors_exist, k, handler='Base', vector_source='default', max_feature_len=1,
                              delete_kid=False):
        """
        Loads a data set, vectorizes it by extracting n-grams (default n=1) using a feature handler (default
        BaseFeatureHandler) and then performs feature selection based on either a vector source or on chi2 scores.
        Returns the encode/decode matrices and the stripped vocabulary of the Vectorizer after feature selection.

        The vector source by default has a unigrams source that covers all unigrams in the training set
        (feature vectors are made up), and does not know about n-grams. Optionally, another vector
        source can be passed in.
        """
        handler_pattern = 'thesisgenerator.plugins.bov_feature_handlers.{}FeatureHandler'
        raw_data, data_ids = load_text_data_into_memory(
            training_path='thesisgenerator/resources/test-tr',
            test_path='thesisgenerator/resources/test-ev',
        )

        tokenizer = _load_tokenizer()
        x_train, y_train, x_test, y_test = tokenize_data(raw_data, tokenizer, data_ids)

        if vector_source == 'default':
            unigrams_vectors = UnigramVectorSource(
                ['thesisgenerator/resources/thesauri/exp0-0a.txt.events-unfiltered.strings'])
            vector_source = CompositeVectorSource([UnigramDummyComposer(unigrams_vectors)],
                                                  0.0, False)

        if delete_kid:
            # the set of vectors we load from disk covers all unigrams in the training set, which makes it boring
            # let's remove one entry
            del unigrams_vectors.entry_index['kid/n']
            unigrams_vectors.feature_matrix = unigrams_vectors.feature_matrix[:, :-1]

        pipeline_list = [
            ('vect',
             ThesaurusVectorizer(min_df=1, vector_source=vector_source, use_tfidf=False,
                                 ngram_range=(1, max_feature_len),
                                 decode_token_handler=handler_pattern.format(handler))),
            ('fs', VectorBackedSelectKBest(vector_source=vector_source,
                                           ensure_vectors_exist=ensure_vectors_exist, k=k)),
            ('dumper', FeatureVectorsCsvDumper('fs-test'))
        ]
        self.p = Pipeline(pipeline_list)

        tr_matrix, tr_voc = self.p.fit_transform(x_train, y_train)
        ev_matrix, ev_voc = self.p.transform(x_test)
        return tr_matrix.A, self._strip(tr_voc), ev_matrix.A, self._strip(ev_voc)

    def test_unigrams_without_feature_selection(self):
        """
        Test without any feature selection and unigram features only, matrices and vocabulary are as in
        test_main.test_baseline_use_all_features_signifier_only.
        """

        # training corpus is "cats like dogs" (x2), "kids play games"
        # eval corpus is "birds like fruit" (x2), "dogs eat birds"
        # thesaurus contains cat, dog, game, kid, fruit, like, play
        tr_matrix, tr_voc, ev_matrix, ev_voc = self._do_feature_selection(False, 'all', vector_source=None)
        # should work without a vector source, because we use signified encoding only and no vector-based FS
        voc = {
            'cat/n': 0,
            'dog/n': 1,
            'game/n': 2,
            'kid/n': 3,
            'like/v': 4,
            'play/v': 5
        }
        self.assertDictEqual(tr_voc, ev_voc)
        self.assertDictEqual(tr_voc, voc)

        t.assert_array_equal(tr_matrix, np.array(
            [[1., 1., 0., 0., 1., 0.],
             [1., 1., 0., 0., 1., 0.],
             [0., 0., 1., 1., 0., 1.]]))
        t.assert_array_equal(ev_matrix, np.array(
            [[0., 0., 0., 0., 1., 0.],
             [0., 0., 0., 0., 1., 0.],
             [0., 1., 0., 0., 0., 0.]]))
        self._check_debug_file(ev_matrix, tr_matrix, voc)


    def test_with_thesaurus_feature_selection_only(self):
        """
        Tests if features in the training data not contained in the vector source are removed. A feature is
        removed from the default vector source.
        """
        tr_matrix, tr_voc, ev_matrix, ev_voc = self._do_feature_selection(True, 'all', delete_kid=True)

        voc = {
            'cat/n': 0,
            'dog/n': 1,
            'game/n': 2,
            #'kid/n': 3, # removed because vector is missing, this happens in self._do_feature_selection
            'like/v': 3,
            'play/v': 4
        }
        self.assertDictEqual(tr_voc, voc)
        self.assertDictEqual(tr_voc, ev_voc)

        t.assert_array_equal(tr_matrix,
                             np.array(
                                 [[1., 1., 0., 1., 0.],
                                  [1., 1., 0., 1., 0.],
                                  [0., 0., 1., 0., 1.]]))
        t.assert_array_equal(ev_matrix,
                             np.array(
                                 [[0., 0., 0., 1., 0.],
                                  [0., 0., 0., 1., 0.],
                                  [0., 1., 0., 0., 0.]]))
        self._check_debug_file(ev_matrix, tr_matrix, voc)


    def test_unigrams_with_chi2_feature_selection_only(self):
        """
        Test the textbook case of feature selection, where some number of features are removed because they are
        not informative. Unigram features only, no vector source needed.
        """
        tr_matrix, tr_voc, ev_matrix, ev_voc = self._do_feature_selection(False, 3, vector_source=None)

        # feature scores at train time are [ 1.  1.  2.  2.  1.  2.]. These are provided by sklearn and I have not
        # verified them. Higher seems to be better (the textbook implementation of chi2 says lower is better)
        voc = {
            #'cat/n': 0, # removed because their chi2 score is low
            #'dog/n': 1,
            'game/n': 0,
            'kid/n': 1,
            #'like/v': 4,
            'play/v': 2
        }
        self.assertDictEqual(tr_voc, voc)
        self.assertDictEqual(tr_voc, ev_voc)

        t.assert_array_equal(tr_matrix,
                             np.array([[0., 0., 0.],
                                       [0., 0., 0.],
                                       [1., 1., 1.]]))

        t.assert_array_equal(ev_matrix,
                             np.array([[0., 0., 0.],
                                       [0., 0., 0.],
                                       [0., 0., 0.]]))
        self._check_debug_file(ev_matrix, tr_matrix, voc)


    def test_with_chi2_and_thesaurus_feature_selection(self):
        """
        Test a combination of feature selection through vector source and low chi2 score. Unigram features only.
        """
        tr_matrix, tr_voc, ev_matrix, ev_voc = self._do_feature_selection(True, 2, delete_kid=True)

        self.assertDictEqual(tr_voc, ev_voc)
        voc = {
            #'cat/n': 0, # removed because of low chi2 score
            #'dog/n': 1,  # removed because of low chi2 score
            'game/n': 0,
            #'kid/n': 3, # removed because vector is missing
            #'like/v': 4,  # removed because of low chi2 score
            'play/v': 1
        }
        # feature scores at train time are [ 1.  1.  2.  2.  1.  2.]
        self.assertDictEqual(tr_voc, voc)

        t.assert_array_equal(tr_matrix, np.array([[0., 0.],
                                                  [0., 0.],
                                                  [1., 1.]]))

        t.assert_array_equal(ev_matrix, np.array([[0., 0.],
                                                  [0., 0.],
                                                  [0., 0.]]))

        self._check_debug_file(ev_matrix, tr_matrix, voc)

    def test_simple_bigram_features_without_fs(self):
        """
        A standard textbook setup with a limited number of useful bigram features, no feature selection of the basis
        of vectors. No vector source needed.
        """
        tr_matrix, tr_voc, ev_matrix, ev_voc = self._do_feature_selection(False, 'all', vector_source=None,
                                                                          max_feature_len=2)
        self.assertTupleEqual(ev_matrix.shape, (3, 10))

        # vocabulary sorted by feature length and then alphabetically-- default behaviour of python's sorted()
        self.assertDictEqual(tr_voc,
                             {'cat/n': 0,
                              'dog/n': 1,
                              'game/n': 2,
                              'kid/n': 3,
                              'like/v': 4,
                              'play/v': 5,
                              'cat/n like/v': 6,
                              'kid/n play/v': 7,
                              'like/v dog/n': 8,
                              'play/v game/n': 9})

        t.assert_array_equal(tr_matrix, self.training_matrix_signifier_bigrams)

    def test_simple_bigram_features_with_chi2_fs(self):
        """
        A standard textbook setup with a limited number of useful bigram features, chi2 feature selection. No
        vector source needed.
        """
        tr_matrix, tr_voc, ev_matrix, ev_voc = self._do_feature_selection(False, 5, vector_source=None,
                                                                          max_feature_len=2)
        self.assertTupleEqual(ev_matrix.shape, (3, 5))

        # feature scores are [ 1.  1.  2.  2.  1.  2.  1.  2.  1.  2.]
        self.assertDictEqual(tr_voc,
                             {#'cat/n': 0, # removed because of low chi2-score
                              #'dog/n': 1,
                              'game/n': 0,
                              'kid/n': 1,
                              #'like/v': 4,
                              'play/v': 2,
                              #'cat/n like/v': 6,
                              'kid/n play/v': 3,
                              #'like/v dog/n': 8,
                              'play/v game/n': 4})
        t.assert_array_equal(tr_matrix, np.array([[0., 0., 0., 0., 0.],
                                                  [0., 0., 0., 0., 0.],
                                                  [1., 1., 1., 1., 1.]]))

    def test_bigram_features_with_composer_without_fs(self):
        """
        A test with all uni- and bi-gram features and a simple predefined vector source for these bigrams. Feature
        handler is SignifierSignifier to excercise nearest-neighbours look-up in the vector source
        """

        # load a mock unigram thesaurus, bypassing the similarity calculation provided by CompositeVectorSource
        composer = PrecomputedSimilaritiesVectorSource(['thesisgenerator/resources/exp0-0a.strings'])

        # patch it to ensure it contains some bigram entries, as if they were calculated on the fly
        composer.th['like/v fruit/n'] = [('like/v', 0.8)]
        tr_matrix, tr_voc, ev_matrix, ev_voc = self._do_feature_selection(False, 'all', handler='SignifierSignified',
                                                                          vector_source=composer, max_feature_len=2)
        self.assertTupleEqual(ev_matrix.shape, (3, 10))
        t.assert_array_equal(tr_matrix, self.training_matrix_signifier_bigrams)

        # vector store says: fruit -> cat 0.06, like fruit -> like 0.8
        ev_expected = np.array([[0.06, 0., 0., 0., 1.8, 0., 0., 0., 0., 0.],
                                [0.06, 0., 0., 0., 1.8, 0., 0., 0., 0., 0.],
                                [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]])

        t.assert_array_equal(ev_expected, ev_matrix)

    def _check_debug_file(self, ev_matrix, tr_matrix, voc):
        for name, matrix in zip(['tr', 'ev'], [tr_matrix, ev_matrix]):
            filename = "PostVectDump_fs-test_%s-cl0-fold'NONE'.csv" % name
            df = read_csv(filename)
            # the columns are u'id', u'target', u'total_feat_weight', u'nonzero_feats', followed by feature vectors
            # check that we have the right number of columns
            self.assertEquals(len(df.columns), 4 + len(voc))
            # check that column names match the vocabulary (after stripping feature metadata)
            self.assertDictEqual(voc, self._strip({eval(v): i for i, v in enumerate(df.columns[4:])}))
            #check that feature vectors are written correctly
            t.assert_array_equal(matrix, df.ix[:, 4:].as_matrix())
            os.remove(filename)

