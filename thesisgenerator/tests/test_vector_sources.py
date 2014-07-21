from itertools import combinations
from unittest import TestCase

from mock import Mock
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix, issparse
import pytest

from thesisgenerator.composers.vectorstore import *
from discoutils.tokens import DocumentFeature, Token


DIM = 10

path = 'thesisgenerator/resources/thesauri/small.txt.events.strings'
unigram_feature = DocumentFeature('1-GRAM', (Token('a', 'N'),))
unk_unigram_feature = DocumentFeature('1-GRAM', ((Token('unk', 'UNK')),))
bigram_feature = DocumentFeature('2-GRAM', (Token('a', 'N'), Token('b', 'V')))
unk_bigram_feature = DocumentFeature('2-GRAM', (Token('a', 'N'), Token('UNK', 'UNK')))
an_feature = DocumentFeature('AN', (Token('c', 'J'), Token('a', 'n')))
known_features = set([unigram_feature, bigram_feature, an_feature])
all_features = set([unigram_feature, bigram_feature, an_feature, unk_unigram_feature, unk_bigram_feature])


class TestUnigramVectorSource(TestCase):
    def setUp(self):
        self.source = Vectors.from_tsv(path)

    def test_get_vector(self):
        # vectors come out right
        # a/N	amod:c	2   T:t1	1	T:t2	2	T:t3	3
        assert_array_equal(
            self.source.get_vector('a/N').todense(),
            [[0., 2., 0., 1., 2., 3., 0., 0.]]
        )

        # vocab is in sorted order
        self.assertListEqual(
            ['also/RB', 'amod:c', 'now/RB', 't:t1', 't:t2', 't:t3', 't:t4', 't:t5', ],
            self.source.columns)

        self.assertIsNone(self.source.get_vector('jfhjgjdfyjhgb/N'))
        self.assertIsNone(self.source.get_vector('jfhjgjdfyjhgb/J'))

    def test_contains(self):
        """
        Test if the unigram model only accepts unigram features
        """
        # for thing in (known_features | unk_unigram_feature | unk_bigram_feature):
        self.assertIn(unigram_feature.tokens_as_str(), self.source)
        for thing in (unk_unigram_feature, bigram_feature, unk_unigram_feature):
            self.assertNotIn(thing.tokens_as_str(), self.source)


class TestAdditiveVectorSource(TestCase):
    def setUp(self):
        unigrams_vectors = Vectors.from_tsv(path)
        self.composer = AdditiveComposer(unigrams_vectors)

    def test_contains(self):
        self.assertIn(bigram_feature.tokens_as_str(), self.composer)
        for s in ['b/V_c/J', 'a/N_c/J', 'b/V_b/V_b/V']:
            self.assertIn((s), self.composer)

        self.assertNotIn(unigram_feature, self.composer)
        self.assertNotIn(unk_unigram_feature, self.composer)
        self.assertNotIn(unk_bigram_feature, self.composer)


class TestSimpleComposers(TestCase):
    def setUp(self):
        self.m = Mock()
        self.m.get_vector.return_value = csr_matrix(np.arange(DIM))

    def test_with_real_data(self):
        source = Vectors.from_tsv(path)
        add = AdditiveComposer(source)
        mult = MultiplicativeComposer(source)

        assert_array_equal(
            np.array([[0, 0, 0, 0, 0, 9, 0, 0]]),
            mult.get_vector('a/N_b/V').A
        )

        assert_array_equal(
            np.array([[5, 2, 7, 1, 2, 6, 0, 0]]),
            add.get_vector('a/N_b/V').A
        )

        assert_array_equal(
            np.array([[5, 11, 15, 1, 2, 6, 10, 4]]),
            add.get_vector('a/N_b/V_c/J').A
        )

    def test_compose(self):
        add = AdditiveComposer(self.m)
        mult = MultiplicativeComposer(self.m)

        for i in range(1, 4):
            print(i)
            df = '_'.join(['a/N'] * i)
            result = add.get_vector(df)
            self.assertTrue(issparse(result))
            assert_array_equal(
                np.arange(DIM).reshape((1, DIM)) * i,
                result.A
            )

            result = mult.get_vector(df)
            self.assertTrue(issparse(result))
            assert_array_equal(
                np.arange(DIM).reshape((1, DIM)) ** i,
                result.A
            )


class TestMinMaxComposer(TestCase):
    def setUp(self):
        self.unigrams_vectors = Vectors.from_tsv('thesisgenerator/resources/ones.vectors.txt')
        self.min_composer = MinComposer(self.unigrams_vectors)
        print(self.unigrams_vectors)
        self.max_composer = MaxComposer(self.unigrams_vectors)

    def test_compose(self):
        f1 = 'a/N_b/V_c/J'
        f2 = 'b/V_c/J'
        f3 = 'b/V'

        assert_array_equal(self.min_composer.get_vector(f1).A.ravel(),
                           np.array([0., 0., 0., 0.]))
        assert_array_equal(self.max_composer.get_vector(f1).A.ravel(),
                           np.array([1., 1., 1., 0.]))

        assert_array_equal(self.min_composer.get_vector(f2).A.ravel(),
                           np.array([0, 0, 0, 0]))
        assert_array_equal(self.max_composer.get_vector(f2).A.ravel(),
                           np.array([0, 1, 1, 0]))

        assert_array_equal(self.min_composer.get_vector(f3).A.ravel(),
                           np.array([0., 1., 0., 0.]))

    def test_contains(self):
        self.assertIn('a/N_b/V_c/J', self.max_composer)
        self.assertIn('b/V_c/J', self.min_composer)
        self.assertNotIn('b/V_c/J_x/N', self.min_composer)
        self.assertNotIn('b/X_c/X', self.min_composer)


class TestHeadAndTailWordComposers(object):
    @pytest.fixture
    def composers(self):
        unigram_vectors = Vectors.from_tsv('thesisgenerator/resources/exp0-0a.strings')

        return LeftmostWordComposer(unigram_vectors), RightmostWordComposer(unigram_vectors)

    def test_contains(self, composers):
        head, tail = composers
        for c in composers:
            for f in ['like/V_fruit/N', 'fruit/N_cat/N', 'kid/N_like/V_fruit/N']:
                assert f in c

        assert 'cat/N' not in head  # no unigrams
        assert 'cat/N' not in tail  # no unigrams
        assert 'red/J_cat/N' not in head  # no unknown head words
        assert 'red/J_cat/N' in tail  # no unknown head words

    def test_get_vector(self, composers):
        head, tail = composers
        v1 = head.get_vector('like/V_fruit/N')
        v2 = tail.get_vector('like/V_fruit/N')
        assert v1.shape == v2.shape == (1, 7)
        assert_array_equal(v1.A.ravel(), np.array([0, 0, 0, 0, 0, 0, 0.11]))
        assert_array_equal(v2.A.ravel(), np.array([0.06, 0.05, 0, 0, 0, 0, 0]))

    def test_compose_all(self, composers):
        composer, _ = composers
        original_matrix, original_cols, original_rows = composer.unigram_source.to_sparse_matrix()
        matrix, cols, rows = composer.compose_all(['cat/N_game/N', DocumentFeature.from_string('dog/N_game/N')])

        # the columns should remain unchanges
        assert original_cols == cols
        # the first rows are for the unigrams that existed before composition- 7 of them
        assert_array_equal(original_matrix.A, matrix.A[:7, :])
        # two new rows should appear, one for each composed feature
        # this should be reflected in both the index and the matrix
        assert rows['cat/N_game/N'] == 7
        assert rows['dog/N_game/N'] == 8
        assert matrix.shape == (9, 7) == (len(rows), len(cols))
        assert_array_equal(matrix.A[7, :], composer.unigram_source.get_vector('cat/N').A.ravel())
        assert_array_equal(matrix.A[8, :], composer.unigram_source.get_vector('dog/N').A.ravel())

