import os
from unittest import TestCase

from sklearn.pipeline import Pipeline
import numpy as np
import numpy.testing as t
from pandas.io.parsers import read_csv

from thesisgenerator.composers.feature_selectors import VectorBackedSelectKBest
from thesisgenerator.composers.vectorstore import CompositeVectorSource, UnigramVectorSource, \
    AdditiveComposer, UnigramDummyComposer
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

    def _do_feature_selection(self, ensure_vectors_exist, k, use_composer=False):
        """
        Loads a data set, vectorizes it using BaseFeatureHandler and then performs feature selection based on
        either a vector source or on chi2 scores. Returns the encode/decode matrices and the vocabulary of the
        Vectorizer after feature selection is done._strip

         The vector source utilised contains all unigram in the training set (feature vectors are made up)
        """
        #Use composer should not make a difference when the feature handler is BaseFeatureHandler at both encode
        #and decode time.
        raw_data, data_ids = load_text_data_into_memory(
            training_path='thesisgenerator/resources/test-tr',
            test_path='thesisgenerator/resources/test-ev',
        )

        tokenizer = _load_tokenizer()
        x_train, y_train, x_test, y_test = tokenize_data(raw_data, tokenizer, data_ids)

        unigrams_vectors = UnigramVectorSource(['thesisgenerator/resources/thesauri/exp0-0a.txt.events.strings'])
        if ensure_vectors_exist:
            # the set of vectors we load from disk covers the entire training set, which makes it boring
            # let's remove one entry
            del unigrams_vectors.entry_index['kid/n']
            unigrams_vectors.feature_matrix = unigrams_vectors.feature_matrix[:, :-1]

        dummy_composer = UnigramDummyComposer(unigrams_vectors)
        add_composer = AdditiveComposer(unigrams_vectors)
        composer_list = [dummy_composer, add_composer] if use_composer else [dummy_composer]
        composers = CompositeVectorSource(unigrams_vectors, composer_list, 0.0, False)

        pipeline_list = [
            ('vect', ThesaurusVectorizer(min_df=1, vector_source=composers, use_tfidf=False)),
            ('fs', VectorBackedSelectKBest(vector_source=composers, ensure_vectors_exist=ensure_vectors_exist, k=k)),
            ('dumper', FeatureVectorsCsvDumper('fs-test'))
        ]
        self.p = Pipeline(pipeline_list)

        tr_matrix, tr_voc = self.p.fit_transform(x_train, y_train)
        ev_matrix, ev_voc = self.p.transform(x_test)

        return tr_matrix.A, self._strip(tr_voc), ev_matrix.A, self._strip(ev_voc)

    def test_without_feature_selection(self):
        """
        Test without any feature selection, matrices and vocabulary are as in
        test_main.test_baseline_use_all_features_signifier_only
        """
        # training corpus is "cats like dogs" (x2), "kids play games"
        # eval corpus is "birds like fruit" (x2), "dogs eat birds"
        # thesaurus contains cat, dog, game, kid, fruit, like, play

        for a in [True, False]:
            tr_matrix, tr_voc, ev_matrix, ev_voc = self._do_feature_selection(False, 'all', use_composer=a)
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
        Tests if features in the training data not contained in the vector source are removed
        """
        tr_matrix, tr_voc, ev_matrix, ev_voc = self._do_feature_selection(True, 'all')

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


    def test_with_chi2_feature_selection_only(self):
        """
        Test the normal case of feature selection, where some number of features are removed because they are
        not informative
        """
        tr_matrix, tr_voc, ev_matrix, ev_voc = self._do_feature_selection(False, 3)

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
        self.assertDictEqual(tr_voc,
                             voc)

        self.assertDictEqual(tr_voc, ev_voc)

        t.assert_array_equal(tr_matrix,
                             np.array(
                                 [[0., 0., 0.],
                                  [0., 0., 0.],
                                  [1., 1., 1.]]))

        t.assert_array_equal(ev_matrix,
                             np.array(
                                 [[0., 0., 0.],
                                  [0., 0., 0.],
                                  [0., 0., 0.]]))
        self._check_debug_file(ev_matrix, tr_matrix, voc)


    def test_with_chi2_and_thesaurus_feature_selection(self):
        """
        Test a combination of feature selection through vector source and low chi2 score.
        """
        tr_matrix, tr_voc, ev_matrix, ev_voc = self._do_feature_selection(True, 2)

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

        t.assert_array_equal(tr_matrix, np.array(
            [[0., 0.],
             [0., 0.],
             [1., 1.]]))
        t.assert_array_equal(ev_matrix, np.array(
            [[0., 0.],
             [0., 0.],
             [0., 0.]]))

        self._check_debug_file(ev_matrix, tr_matrix, voc)


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

