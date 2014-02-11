from itertools import combinations
from unittest import TestCase

from mock import Mock
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix, issparse
import pytest

from thesisgenerator.composers.vectorstore import *
from discoutils.tokens import DocumentFeature, Token


DIM = 10

path = ['thesisgenerator/resources/thesauri/small.txt.events.strings']
unigram_feature = DocumentFeature('1-GRAM', (Token('a', 'N'),))
unk_unigram_feature = DocumentFeature('1-GRAM', ((Token('unk', 'UNK')),))
bigram_feature = DocumentFeature('2-GRAM', (Token('a', 'N'), Token('b', 'V')))
unk_bigram_feature = DocumentFeature('2-GRAM', (Token('a', 'N'), Token('UNK', 'UNK')))
an_feature = DocumentFeature('AN', (Token('c', 'J'), Token('a', 'n')))
known_features = set([unigram_feature, bigram_feature, an_feature])
all_features = set([unigram_feature, bigram_feature, an_feature, unk_unigram_feature, unk_bigram_feature])


class TestUnigramVectorSource(TestCase):
    def setUp(self):
        self.source = UnigramVectorSource(path)
        self.reduced_source = UnigramVectorSource(path, reduce_dimensionality=True, dimensions=2)

    def test_get_vector(self):
        # vectors come out right
        # a/N	amod:c	2   T:t1	1	T:t2	2	T:t3	3
        assert_array_equal(
            self.source._get_vector(DocumentFeature.from_string('a/N')).todense(),
            [[0., 2., 0., 1., 2., 3., 0., 0.]]
        )

        # vocab is in sorted order
        self.assertListEqual(
            ['also/RB', 'amod:c', 'now/RB', 't:t1', 't:t2', 't:t3', 't:t4', 't:t5', ],
            self.source.distrib_features_vocab)

        self.assertIsNone(self.source._get_vector(DocumentFeature.from_string('jfhjgjdfyjhgb/N')))
        self.assertIsNone(self.source._get_vector(DocumentFeature.from_string('jfhjgjdfyjhgb/J')))

    def test_contains(self):
        """
        Test if the unigram model only accepts unigram features
        """
        #for thing in (known_features | unk_unigram_feature | unk_bigram_feature):
        self.assertIn(unigram_feature, self.source)
        for thing in (unk_unigram_feature, bigram_feature, unk_unigram_feature):
            self.assertNotIn(thing, self.source)

    def test_dimensionality_reduction(self):
        v = self.reduced_source._get_vector(DocumentFeature.from_string('a/N'))
        self.assertTupleEqual((1, 2), v.shape)
        print v.A


class TestAdditiveVectorSource(TestCase):
    def setUp(self):
        unigrams_vectors = UnigramVectorSource(path)
        self.composer = AdditiveComposer(unigrams_vectors)

    def test_contains(self):
        self.assertIn(bigram_feature, self.composer)
        for s in ['b/V_c/J', 'a/N_c/J', 'b/V_b/V_b/V']:
            self.assertIn(DocumentFeature.from_string(s), self.composer)

        self.assertNotIn(unigram_feature, self.composer)
        self.assertNotIn(unk_unigram_feature, self.composer)
        self.assertNotIn(unk_bigram_feature, self.composer)

    def test_get_nearest_neighbour(self):
        unigrams_vectors = UnigramVectorSource(['thesisgenerator/resources/ones.vectors.txt'])
        composer = CompositeVectorSource([AdditiveComposer(unigrams_vectors)], 0, True)
        tokens_only = [x.tokens[0] for x in unigrams_vectors.entry_index.keys()]
        vocab = [DocumentFeature('2-GRAM', (x, y)) for (x, y) in
                 combinations(tokens_only, 2)]
        composer.populate_vector_space(vocab)
        print '----------- Setting include_self to True ------------'
        for bigram in vocab:
            neighbours = composer.get_nearest_neighbours(bigram)
            self.assertEquals(len(neighbours), 1)
            neighbour, sim = neighbours[0]
            print composer._get_vector(bigram), bigram, neighbours
            self.assertEqual(bigram, neighbour)
            self.assertAlmostEqual(sim, 1, 5)

        print '----------- Setting include_self to False ------------'
        composer.include_self = False
        for bigram in vocab:
            neighbours = composer.get_nearest_neighbours(bigram)
            self.assertEquals(len(neighbours), 1)
            neighbour, sim = neighbours[0]
            print bigram, neighbours
            self.assertNotEqual(bigram, neighbour)
            # The vectors for unigrams are 4D one-hot encoded, ie. a=1000, b=0100,...,d=0001
            # The pointwise sum of any of these two has two ones and two zeros. If v1 = x1+y1, v2=x2+y2,
            # ( x1,x2,y1,y2 \in {a,b,c,d} ) then v1 and v2 share either 0, 1 or 2 non-zero dimensions.
            # If they share 2, i.e. we've added the same terms up and include_self will prevent this neighbour
            # from being returned. If they share no dimensions, the cosine sim is 0 and the sim_threshold will
            # kick in. The nearest neighbour of the sum of any two vectors (not including itself) is the one where
            # they only share one dimension, i.e. [1,0,1,0] and [1,0,0,1], and the cosine of these two is ~0.5.

            self.assertAlmostEqual(sim, 0.5, 5)

        composer.sim_threshold = 0.6
        # no neighbours should be returned now, because 0.6 > 0.5
        for bigram in vocab:
            neighbours = composer.get_nearest_neighbours(bigram)
            self.assertEquals(len(neighbours), 0)


class TestCompositeVectorSource(TestCase):
    def setUp(self):
        unigrams_vectors = UnigramVectorSource(path)
        self.composer = CompositeVectorSource([AdditiveComposer(unigrams_vectors),
                                               UnigramDummyComposer(unigrams_vectors)],
                                              0.0, False)

    def test_entry_index(self):
        unigrams_vectors = UnigramVectorSource(['thesisgenerator/resources/ones.vectors.txt'])
        subcomposers = [AdditiveComposer(unigrams_vectors), MultiplicativeComposer(unigrams_vectors),
                        LeftmostWordComposer(unigrams_vectors), RightmostWordComposer(unigrams_vectors)]
        composer = CompositeVectorSource(subcomposers,
                                         sim_threshold=0, include_self=True)
        tokens_only = [x.tokens[0] for x in unigrams_vectors.entry_index.keys()]
        vocab = [DocumentFeature('2-GRAM', (x, y)) for (x, y) in
                 combinations(tokens_only, 2)]
        composer.populate_vector_space(vocab)

        m = composer.feature_matrix.A
        e = composer.debug_entry_index
        self.assertEqual(len(e), len(vocab) * len(subcomposers))

        for row, (feature, composer) in enumerate(e):
            expected = composer._get_vector(feature).A.ravel()
            observed = m[row, :]
            assert_array_equal(expected, observed)


    def test_contains(self):
        self.assertIn(unigram_feature, self.composer)
        self.assertIn(bigram_feature, self.composer)

        self.assertNotIn(unk_unigram_feature, self.composer)
        self.assertNotIn(unk_bigram_feature, self.composer)

        #def test_only_composable_features(self):
        #    """
        #    Test that a set of composer accepts only things that can be composed
        #    """
        #    source = CompositeVectorSource(self.conf)
        #    self.assertIn(bigram_feature, source)
        #    self.assertIn(an_feature, source)
        #    self.assertNotIn(unk_unigram_feature, source)
        #    self.assertNotIn(unk_unigram_feature, source)
        #    self.assertNotIn(unigram_feature, source)
        #
        #
        #def test_also_include_unigram_features(self):
        #    """
        #    Test that when include_unigram_features is enabled unigram features as well as
        #    composable features are accepted
        #    """
        #    self.conf['include_unigram_features'] = True
        #    source = CompositeVectorSource(self.conf)
        #    for x in known_features:
        #        self.assertIn(x, source)
        #
        #def test_include_unigram_features_only(self):
        #    self.conf['include_unigram_features'] = True
        #    self.conf = {key: value for key, value in self.conf.items() if 'Composer' not in key}
        #    source = CompositeVectorSource(self.conf)
        #    self.assertNotIn(bigram_feature, source)
        #    self.assertIn(unigram_feature, source)
        #    #self.assertSetEqual(source.__contains__(known_features), unigram_feature)

        #def test_build_vector_space(self):
        #    source = CompositeVectorSource(self.conf)
        #    training_features = [x for x in all_features if x in source]
        #    source.populate_vector_space(training_features)
        #    for f in training_features:
        #        #print 'Composing', f
        #        #print 'Composed vectors are ', source.get_vector(f)
        #        #print 'Nearest neighbours are\n'
        #        for (_, (dist, _)) in source._get_nearest_neighbours(f):
        #            # nearest neighbour should be the feature itself
        #            self.assertAlmostEquals(dist, 0., places=4)
        #            #print '---------------------------'
        #
        #    for comp, dist, neigh in source._get_nearest_neighbours(('2-GRAM', ('c/j', 'a/n'))):
        #        self.assertIn(neigh, training_features)
        #        #print comp, dist, neigh
        #        #todo expand this test


class TestSimpleComposers(TestCase):
    def setUp(self):
        self.m = Mock()
        self.m._get_vector.return_value = csr_matrix(np.arange(DIM))

    def test_with_real_data(self):
        source = UnigramVectorSource(path)
        add = AdditiveComposer(source)
        mult = MultiplicativeComposer(source)

        assert_array_equal(
            np.array([[0, 0, 0, 0, 0, 9, 0, 0]]),
            mult._get_vector(DocumentFeature.from_string('a/N_b/V')).A
        )

        assert_array_equal(
            np.array([[5, 2, 7, 1, 2, 6, 0, 0]]),
            add._get_vector(DocumentFeature.from_string('a/N_b/V')).A
        )

        assert_array_equal(
            np.array([[5, 11, 15, 1, 2, 6, 10, 4]]),
            add._get_vector(DocumentFeature.from_string('a/N_b/V_c/J')).A
        )

    def test_compose(self):
        add = AdditiveComposer(self.m)
        mult = MultiplicativeComposer(self.m)

        for i in range(1, 4):
            print i
            df = DocumentFeature.from_string('_'.join(['a/N'] * i))
            result = add._get_vector(df)
            self.assertTrue(issparse(result))
            assert_array_equal(
                np.arange(DIM).reshape((1, DIM)) * i,
                result.A
            )

            result = mult._get_vector(df)
            self.assertTrue(issparse(result))
            assert_array_equal(
                np.arange(DIM).reshape((1, DIM)) ** i,
                result.A
            )


class TestPrecomputedSimSource(TestCase):
    def setUp(self):
        self.source = PrecomputedSimilaritiesVectorSource.from_file(
            thesaurus_files=['thesisgenerator/resources/exp0-0a.strings'],
            sim_threshold=0, include_self=False)
        self.source2 = PrecomputedSimilaritiesVectorSource.from_file(
            thesaurus_files=['thesisgenerator/resources/exp0-0c.strings'],
            sim_threshold=0, include_self=False)


    def test_get_nearest_neighbours(self):
        self.assertTupleEqual(
            self.source.get_nearest_neighbours(DocumentFeature.from_string('cat/N'))[0],
            (DocumentFeature('1-GRAM', (Token('dog', 'N'),)), 0.8)
        )

        self.assertTupleEqual(
            self.source2.get_nearest_neighbours(DocumentFeature.from_string('a/J_b/N'))[0],
            (DocumentFeature.from_string('g/N'), 0.8)
        )

    def test_contains(self):
        self.assertTrue(DocumentFeature.from_string('cat/N') in self.source)
        self.assertFalse(DocumentFeature.from_string('a/J_b/N') in self.source)

        self.assertTrue(DocumentFeature.from_string('a/N') in self.source2)
        self.assertTrue(DocumentFeature.from_string('a/J_b/N') in self.source2)

# commented out because it requires manual setup to binary resources
#class TestBaroniComposer(object):
#    @pytest.fixture
#    def composer(self):
#        unigram_source = UnigramVectorSource(
#            ['thesisgenerator/resources/baroni/julie.onlyN-SVD300.clean.vectors'])
#        model_file = 'thesisgenerator/resources/baroni/julie.ANs.clean.AN-model.model.pkl'
#        return BaroniComposer(unigram_source, model_file)
#
#    def test_contains(self, composer):
#        # that composers only contains african/J and a bunch of nouns
#        assert DocumentFeature.from_string('african/J_price/N') in composer
#        assert DocumentFeature.from_string('african/J_south/N') in composer
#        assert DocumentFeature.from_string('african/J_somemadeupword/N') not in composer

class TestMinMaxComposer(TestCase):
    def setUp(self):
        self.unigrams_vectors = UnigramVectorSource(['thesisgenerator/resources/ones.vectors.txt'])
        self.min_composer = MinComposer(self.unigrams_vectors)
        self.max_composer = MaxComposer(self.unigrams_vectors)

    def test_compose(self):
        f1 = DocumentFeature.from_string('a/N_b/V_c/J')
        f2 = DocumentFeature.from_string('b/V_c/J')
        f3 = DocumentFeature.from_string('b/V')

        assert_array_equal(self.min_composer._get_vector(f1).A.ravel(),
                           np.array([0., 0., 0., 0.]))
        assert_array_equal(self.max_composer._get_vector(f1).A.ravel(),
                           np.array([1., 1., 1., 0.]))

        assert_array_equal(self.min_composer._get_vector(f2).A.ravel(),
                           np.array([0, 0, 0, 0]))
        assert_array_equal(self.max_composer._get_vector(f2).A.ravel(),
                           np.array([0, 1, 1, 0]))

        assert_array_equal(self.min_composer._get_vector(f3).A.ravel(),
                           np.array([0., 1., 0., 0.]))

    def test_contains(self):
        self.assertIn(DocumentFeature.from_string('a/N_b/V_c/J'), self.max_composer)
        self.assertIn(DocumentFeature.from_string('b/V_c/J'), self.min_composer)
        self.assertNotIn(DocumentFeature.from_string('b/V_c/J_x/N'), self.min_composer)
        self.assertNotIn(DocumentFeature.from_string('b/X_c/X'), self.min_composer)


class TestHeadAndTailWordComposers(object):
    @pytest.fixture
    def composers(self):
        unigram_vectors = UnigramVectorSource(['thesisgenerator/resources/exp0-0a.strings'])

        return LeftmostWordComposer(unigram_vectors), RightmostWordComposer(unigram_vectors)

    def test_contains(self, composers):
        head, tail = composers
        for c in composers:
            for f in ['like/V_fruit/N', 'fruit/N_cat/N', 'kid/N_like/V_fruit/N']:
                assert DocumentFeature.from_string(f) in c

        assert DocumentFeature.from_string('cat/N') not in head # no unigrams
        assert DocumentFeature.from_string('cat/N') not in tail # no unigrams
        assert DocumentFeature.from_string('red/J_cat/N') not in head # no unknown head words
        assert DocumentFeature.from_string('red/J_cat/N') in tail # no unknown head words

    def test_get_vector(self, composers):
        head, tail = composers
        v1 = head._get_vector(DocumentFeature.from_string('like/V_fruit/N'))
        v2 = tail._get_vector(DocumentFeature.from_string('like/V_fruit/N'))
        assert v1.shape == v2.shape == (1, 7)
        assert_array_equal(v1.A.ravel(), np.array([0, 0, 0, 0, 0, 0, 0.11]))
        assert_array_equal(v2.A.ravel(), np.array([0.06, 0.05, 0, 0, 0, 0, 0]))
