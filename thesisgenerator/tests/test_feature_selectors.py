from pprint import pprint
from unittest import TestCase

from sklearn.pipeline import Pipeline
import numpy as np
import numpy.testing as t

from thesisgenerator.composers.feature_selectors import VectorBackedSelectKBest
from thesisgenerator.composers.vectorstore import CompositeVectorSource, UnigramVectorSource, \
    AdditiveComposer, UnigramDummyComposer
from thesisgenerator.plugins.bov import ThesaurusVectorizer
from thesisgenerator.plugins.dumpers import FeatureVectorsCsvDumper
from thesisgenerator.utils.data_utils import load_text_data_into_memory, _load_tokenizer, tokenize_data


__author__ = 'mmb28'


class TestVectorBackedSelectKBest(TestCase):
    def setUp(self):
        # training corpus is "cats like dogs" (x2), "kids play games"
        # eval corpus is "birds like fruit" (x2), "dogs eat birds"
        # thesaurus contains cat, dog, game, kid, fruit, like, play
        self.full_vocab = {
            'cat/n': 0,
            'dog/n': 1,
            'game/n': 2,
            'kid/n': 3,
            'like/v': 4,
            'play/v': 5
        }

        # with signifier encoding, only one known feature
        self.full_training_matrix = np.array(
            [[1., 1., 0., 0., 1., 0.],
             [1., 1., 0., 0., 1., 0.],
             [0., 0., 1., 1., 0., 1.]])

        self.full_eval_matrix = np.array(
            [[0., 0., 0., 0., 1., 0.],
             [0., 0., 0., 0., 1., 0.],
             [0., 1., 0., 0., 0., 0.]])

    def _strip(self, mydict):
        #{ ('1-GRAM', ('X',)) : int} -> {'X' : int}
        for k, v in mydict.iteritems():
            self.assertEquals(len(k), 2)
        return {k[1][0]: v for k, v in mydict.iteritems()}

    def _run(self, ensure_vectors_exist, k, use_composer=False):
        """
        Use composer should not make a difference when the feature handler is BaseFeatureHandler at both encode
        and decode time.
        """
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
            ('vect', ThesaurusVectorizer()),
            ('fs', VectorBackedSelectKBest(vector_source=composers, ensure_vectors_exist=ensure_vectors_exist, k=k)),
            ('dumper', FeatureVectorsCsvDumper('fs-test'))
        ]
        self.p = Pipeline(pipeline_list)

        call_args = {
            'vect__vector_source': composers,
            'vect__use_tfidf': False,
            'vect__min_df': 1,
        }
        self.p.set_params(**call_args)
        tr_matrix, tr_voc = self.p.fit_transform(x_train, y_train)
        #print tr_matrix.A
        #pprint(sorted(self._strip(tr_voc).items()))
        ev_matrix, ev_voc = self.p.transform(x_test)
        #print ev_matrix.A
        pprint(sorted(self._strip(ev_voc).items()))

        return tr_matrix.A, self._strip(tr_voc), ev_matrix.A, self._strip(ev_voc)

    def test_without_feature_selection(self):
        for a in [True, False]:
            tr_matrix, tr_voc, ev_matrix, ev_voc = self._run(False, 'all', use_composer=a)

            self.assertDictEqual(tr_voc, ev_voc)
            self.assertDictEqual(tr_voc, self.full_vocab)

            t.assert_array_equal(tr_matrix, self.full_training_matrix)
            t.assert_array_equal(ev_matrix, self.full_eval_matrix)
            # todo check the resultant debug file on disk, the header must be correct and should match column contents

    def test_with_thesaurus_feature_selection_only(self):
        #todo this is just copied from above, should fail
        tr_matrix, tr_voc, ev_matrix, ev_voc = self._run(True, 'all', use_composer=False)

        self.assertDictEqual(tr_voc, ev_voc)
        self.assertDictEqual(tr_voc, self.full_vocab)

        t.assert_array_equal(tr_matrix, self.full_training_matrix)
        t.assert_array_equal(ev_matrix, self.full_eval_matrix)
        # todo check the resultant debug file on disk, the header must be correct and should match column contents