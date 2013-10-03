from unittest import TestCase
from mock import Mock
import numpy as np

from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix, issparse

from thesisgenerator.composers.vectorstore import UnigramVectorSource, CompositeVectorSource, AdditiveComposer, MultiplicativeComposer

DIM = 10

path = ['thesisgenerator/resources/vectors/small.strings']
unigram_feature = {('1-GRAM', ('a/n',))}
unk_unigram_feature = {('1-GRAM', ('UNK/UNK',))}
bigram_feature = {('2-GRAM', ('a/n', 'b/v'))}
unk_bigram_feature = {('2-GRAM', ('a/n', 'UNK/UNK'))}
an_feature = {('AN', ('c/j', 'a/n'))}
known_features = unigram_feature | bigram_feature | an_feature
all_features = known_features | unk_unigram_feature | unk_bigram_feature


class TestUnigramVectorSource(TestCase):
    def setUp(self):
        self.source = UnigramVectorSource(path)

    def test_get_vector(self):
        # vectors come out right
        # a/N	amod:c	2   T:t1	1	T:t2	2	T:t3	3
        assert_array_equal(
            self.source.get_vector('a/n').todense(),
            [[0., 2., 0., 1., 2., 3., 0., 0.]]
        )

        # vocab is in sorted order
        self.assertDictEqual(
            {'also/rb': 0,
             'amod:c': 1,
             'now/rb': 2,
             't:t1': 3,
             't:t2': 4,
             't:t3': 5,
             't:t4': 6,
             't:t5': 7,
            },
            self.source.distrib_features_vocab)

        self.assertIsNone(self.source.get_vector('jfhjgjdfyjhgb'))

    def test_accept_features(self):
        """
        Test if the unigram model only accepts unigram features
        """
        self.assertSetEqual(
            self.source.accept_features(known_features | unk_unigram_feature | unk_bigram_feature),
            unigram_feature
        )


class TestCompositeVectorSource(TestCase):
    def setUp(self):
        self.conf = {
            'unigram_paths': path,
            'include_unigram_features': False,
            'thesisgenerator.composers.vectorstore.AdditiveComposer': {'run': True},
            'thesisgenerator.composers.vectorstore.MultiplicativeComposer': {'run': True},
            'thesisgenerator.composers.vectorstore.BaroniComposer': {
                'run': True,
                'file_path': '/some/path'}
        }


    def test_only_composable_features(self):
        """
        Test that a set of composer accepts only things that can be composed
        """
        source = CompositeVectorSource(self.conf)

        print source.accept_features(all_features)
        self.assertSetEqual(
            source.accept_features(all_features),
            known_features - unigram_feature
        )

    def test_also_include_unigram_features(self):
        """
        Test that when include_unigram_features is enabled unigram features as well as
        composable features are accepted
        """
        self.conf['include_unigram_features'] = True
        source = CompositeVectorSource(self.conf)
        self.assertSetEqual(source.accept_features(known_features), known_features)

    def test_include_unigram_features_only(self):
        self.conf['include_unigram_features'] = True
        self.conf = {key: value for key, value in self.conf.items() if 'Composer' not in key}
        source = CompositeVectorSource(self.conf)
        self.assertSetEqual(source.accept_features(known_features), unigram_feature)

    def test_build_vector_space(self):
        source = CompositeVectorSource(self.conf)
        training_features = source.accept_features(all_features)
        source.populate_vector_space(training_features)
        for f in training_features:
            #print 'Composing', f
            #print 'Composed vectors are ', source.get_vector(f)
            #print 'Nearest neighbours are\n'
            for (_, dist, _) in source.get_nearest_neighbours(f):
                # nearest neighbour should be the feature itself
                self.assertAlmostEquals(dist, 0., places=4)
                #print '---------------------------'

        for comp, dist, neigh in source.get_nearest_neighbours(('2-GRAM', ('c/j', 'a/n'))):
            self.assertIn(neigh, training_features)
            #print comp, dist, neigh
            #todo expand this test


class TestSimpleComposers(TestCase):
    def setUp(self):
        self.m = Mock()
        self.m.get_vector.return_value = csr_matrix(np.arange(DIM))

    def test_with_real_data(self):
        source = UnigramVectorSource(path)
        add = AdditiveComposer(source)
        mult = MultiplicativeComposer(source)

        assert_array_equal(
            np.array([[0, 0, 0, 0, 0, 9, 0, 0]]),
            mult.get_vector(['a/n', 'b/v']).A
        )

        assert_array_equal(
            np.array([[5, 2, 7, 1, 2, 6, 0, 0]]),
            add.get_vector(['a/n', 'b/v']).A
        )

        assert_array_equal(
            np.array([[5, 11, 15, 1, 2, 6, 10, 4]]),
            add.get_vector(['a/n', 'b/v', 'c/j']).A
        )

    def test_compose(self):
        add = AdditiveComposer(self.m)
        mult = MultiplicativeComposer(self.m)

        for i in np.arange(1, DIM):
            print i
            result = add.get_vector(['a'] * i)
            self.assertTrue(issparse(result))
            assert_array_equal(
                np.arange(DIM).reshape((1, DIM)) * i,
                result.A
            )

            result = mult.get_vector(['a'] * i)
            self.assertTrue(issparse(result))
            assert_array_equal(
                np.arange(DIM).reshape((1, DIM)) ** i,
                result.A
            )