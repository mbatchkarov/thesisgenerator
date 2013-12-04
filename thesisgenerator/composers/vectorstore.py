from abc import ABCMeta, abstractmethod
from collections import defaultdict
import logging
from random import choice
from operator import itemgetter
from pickle import load

from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
from scipy.sparse import vstack
from sklearn.random_projection import SparseRandomProjection

from thesisgenerator.composers import utils
from thesisgenerator.plugins.thesaurus_loader import Thesaurus
from thesisgenerator.plugins.tokenizers import DocumentFeature, Token

import numpy as np


class VectorSource(object):
    __metaclass__ = ABCMeta

    feature_pattern = {}    # each VectorSource can work with a set of feature types

    @abstractmethod
    def __contains__(self, feature):
        """
        Filters out document features that cannot be handled by the implementing model. For instance,
        BaroniComposer cannot handle noun compounds or AN compounds for some adjectives. Features
        are assumed to be generated externally
        """
        pass

    @abstractmethod
    def _get_vector(self, tokens):
        pass


class UnigramVectorSource(VectorSource):
    """
    Holds vectors for a bunch of unigrams.
    """
    feature_pattern = {'1-GRAM'}
    name = 'Lex'

    def __init__(self, unigram_paths, reduce_dimensionality=False, dimensions=1000):
        if not unigram_paths:
            raise ValueError('You must provide a unigram vector file')

        thesaurus = Thesaurus(
            thesaurus_files=unigram_paths,
            sim_threshold=0,
            include_self=False,
            aggressive_lowercasing=False)

        # distributional features of each unigram in the loaded file
        self.feature_matrix, self.distrib_features_vocab, _ = thesaurus.to_sparse_matrix()
        # todo optimise- call below will invoke to_sparse_matrix again. to_core_space should be a classmethod

        # todo it's a bit silly to hold a dissect space as well as a Thesaurus object
        self.dissect_core_space = thesaurus.to_dissect_core_space()

        # Token -> row number in self.feature_matrix that holds corresponding vector
        self.entry_index = {DocumentFeature.from_string(feature): i for (i, feature) in enumerate(thesaurus.keys())}

        assert len(self.entry_index) == len(self.dissect_core_space.id2row)

        # the pos of all unigrams, the type of all n-grams
        self.available_pos = set(feature.tokens[0].pos if feature.type == '1-GRAM' else feature.type
                                 for feature in self.entry_index.keys())
        if reduce_dimensionality:
            logging.info('Reducing dimensionality of unigram vectors from %s to %s',
                         self.feature_matrix.shape[1], dimensions)
            self.transformer = SparseRandomProjection(n_components=dimensions)
            self.feature_matrix = self.transformer.fit_transform(self.feature_matrix)
            self.distrib_features_vocab = None


    def _get_vector(self, feature):
        # word must be an iterable of Token objects
        """
        Returns a matrix of size (1, N) for the first token in the provided list. Warns if multiple tokens are given.
        :param feature: a list of tokens to get vector for
        :rtype: scipy.sparse.csr_matrix
        """
        try:
            row = self.entry_index[feature]
            if len(feature.tokens) > 1:
                logging.warn('Attempting to get unigram vector of n-gram %r', feature)
        except KeyError:
            return None
        return self.feature_matrix[row, :]

    def __contains__(self, feature):
        """
        Accept all unigrams that we have a vector for
        the thing is a unigram and we have a corpus-based vector for that unigram
        """
        return feature.type in self.feature_pattern and feature in self.entry_index

    def __str__(self):
        return '[UnigramVectorSource with %d %d-dimensional entries]' % self.feature_matrix.shape

    def __len__(self):
        return len(self.entry_index)


class Composer(VectorSource):
    def __init__(self, unigram_source=None):
        if not unigram_source:
            raise ValueError('Composers need a unigram vector source')
        self.unigram_source = unigram_source

    def __repr__(self):
        return self.__str__()


class UnigramDummyComposer(Composer):
    """
    This is different from PrecomputedSimilaritiesVectorSource as it contains vectors for each entry and NOT a
    precomputed all-pairs similarity matrix
    """
    name = 'Lexical'
    feature_pattern = {'1-GRAM'}

    def __init__(self, unigram_source=None):
        super(UnigramDummyComposer, self).__init__(unigram_source)

    def __contains__(self, feature):
        return feature in self.unigram_source

    def _get_vector(self, feature):
        return self.unigram_source._get_vector(feature)

    def __str__(self):
        return '[UnigramDummyComposer wrapping %s]' % self.unigram_source


#class OxfordSvoComposer(Composer):
#    name = 'DummySVO'
#    feature_pattern = {'SVO'}
#
#    def __init__(self, unigram_source=None):
#        super(OxfordSvoComposer, self).__init__(unigram_source)
#        if 'V' not in unigram_source.available_pos or \
#                        'N' not in unigram_source.available_pos:
#            raise ValueError('This composer requires a noun and verb unigram vector sources')
#
#    def __contains__(self, feature):
#        """
#        Accept all subject-verb-object phrases where we have a corpus-observed vector for each unigram
#        """
#        if feature.type not in self.feature_pattern:
#            # ignore non-SVO features
#            return False
#
#        for token in feature.tokens:
#            if DocumentFeature('1-GRAM', (token,)) not in self.unigram_source:
#                # ignore ANs containing unknown nouns
#                return False
#
#        return True
#
#    def _get_vector(self, tokens):
#        return self.unigram_source._get_vector((tokens[1],))


class AdditiveComposer(Composer):
    name = 'Add'
    # composers in general work with n-grams (for simplicity n<4)
    feature_pattern = {'2-GRAM', '3-GRAM', 'AN', 'NN', 'VO', 'SVO'}

    def __init__(self, unigram_source=None):
        super(AdditiveComposer, self).__init__(unigram_source)
        self.function = np.add

    def _get_vector(self, feature):
        return sp.csr_matrix(reduce(self.function, [self.unigram_source._get_vector(t).A for t in feature[:]]))


    def __contains__(self, feature):
        """
        Contains all sequences of words where we have a distrib vector for each unigram
        they contain. Rejects unigrams.
        """
        if feature.type not in self.feature_pattern:
            # no point in trying
            return False
        return all(f in self.unigram_source for f in feature[:])

    def __str__(self):
        return '[%s with %d unigram entries]' % (self.__class__.__name__, len(self.unigram_source))


class MultiplicativeComposer(AdditiveComposer):
    name = 'Mult'

    def __init__(self, unigram_source=None):
        super(MultiplicativeComposer, self).__init__(unigram_source)
        self.function = np.multiply


class MinComposer(MultiplicativeComposer):
    name = 'Min'

    def __init__(self, unigram_source=None):
        super(MinComposer, self).__init__(unigram_source)
        self.function = lambda m, n: np.minimum(m, n)


class MaxComposer(MinComposer):
    name = 'Max'

    def __init__(self, unigram_source=None):
        super(MaxComposer, self).__init__(unigram_source)
        self.function = lambda m, n: np.maximum(m, n)


class HeadWordComposer(AdditiveComposer):
    name = 'Head'

    def __init__(self, unigram_source=None):
        super(HeadWordComposer, self).__init__(unigram_source)
        self.hardcoded_index = 0
        self.feature_pattern = {'2-GRAM', '3-GRAM', 'AN', 'NN', 'VO', 'SVO'}


    def _get_vector(self, feature):
        return self.unigram_source._get_vector(feature[self.hardcoded_index])

    def __contains__(self, feature):
        if feature.type not in self.feature_pattern:
            # no point in composing single-word document features
            return False

        return feature[self.hardcoded_index] in self.unigram_source


class TailWordComposer(HeadWordComposer):
    name = 'Tail'

    def __init__(self, unigram_source=None):
        super(TailWordComposer, self).__init__(unigram_source)
        self.hardcoded_index = -1


class BaroniComposer(Composer):
    # BaroniComposer composes AN features
    feature_pattern = {'AN', 'NN'}
    name = 'Baroni'

    def __init__(self, unigram_source=None, pretrained_model_file=None):
        super(BaroniComposer, self).__init__(unigram_source)
        if not pretrained_model_file:
            logging.error('Expected filename, got %s', pretrained_model_file)
            raise ValueError('Model file required to perform composition.')
        with open(pretrained_model_file) as infile:
            self.composer = load(infile)

        # verify the composer's internal structure matches the unigram source
        self.available_modifiers = set(self.composer.function_space.id2row)
        features = self.composer.composed_id2column
        dimensionality = len(self.composer.composed_id2column)

        assert unigram_source.distrib_features_vocab == self.composer.composed_id2column
        self.dissect_core_space = unigram_source.dissect_core_space

        # check composed space's columns matche core space's (=unigram source)'s columns
        assert self.dissect_core_space.id2column == self.composer.composed_id2column

        if 'N' not in unigram_source.available_pos:
            raise ValueError('This composer requires a noun unigram vector source')

            #vector = self._get_vector(DocumentFeature.from_string('african/J_police/N'))
            #logging.info(vector)

    def __contains__(self, feature):
        """
        Accept all adjective-noun or noun-noun phrases where we have a corpus-observed vector for the head and
        a learnt matrix (through PLSR) for the modifier
        """
        # todo expand unit tests now that we have a real composer
        if feature.type not in self.feature_pattern:
            # ignore non-AN features
            #print "%s is not a valid type" % feature.type
            return False

        modifier, head = feature.tokens
        assert ('J', 'N') == (modifier.pos, head.pos) or ('N', 'N') == (modifier.pos, head.pos)

        #if DocumentFeature('1-GRAM', (noun,)) not in self.unigram_source:
        if DocumentFeature.from_string(str(head)) not in self.unigram_source:
            # ignore ANs containing unknown nouns
            # this implementation saves a bit of work by not calling UnigramSource__contains__
            #print "%s not in entry index" % head
            return False

        #if str(modifier) not in self.available_modifiers:
        #    print "%s not in available modifiers" % modifier

        # ignore ANs containing unknown adjectives
        return str(modifier) in self.available_modifiers

    def _get_vector(self, feature):
        #todo test properly
        """

        :param feature: DocumentFeature to compose, assumed to be an adjective/noun and a noun, with PoS tags
        :return:
         :rtype: 1xN scipy sparse matrix of type numpy.float64 with M stored elements in Compressed Sparse Row format,
         where N is the dimensionality of the vectors in the unigram source
        """
        modifier = str(feature.tokens[0])
        head = str(feature.tokens[1])
        phrase = '{}_{}'.format(modifier, head)
        x = self.composer.compose([(modifier, head, phrase)], self.dissect_core_space).cooccurrence_matrix.mat
        #todo could also convert to dense 1D ndarray, vector.A.ravel()
        return x
        #return self.unigram_source._get_vector((feature[-1], ))


class CompositeVectorSource(VectorSource):
    """
    An object that takes vectors and composers as parameters and computes nearest neighbours on the fly
    """
    name = 'Composite'

    def __init__(self, composers, sim_threshold, include_self):
        self.composers = composers
        self.sim_threshold = sim_threshold
        self.include_self = include_self

        self.nbrs, self.feature_matrix, entry_index = [None] * 3     # computed by self.build_peripheral_space()
        self.composers = composers
        self.composer_mapping = defaultdict(set) # feature type -> {composer object}
        #tmp = OrderedDict() # see below
        for c in self.composers:
            for p in c.feature_pattern:
                self.composer_mapping[p].add(c)
                #self.composer_mapping.update(tmp)

    def __contains__(self, feature):
        return any([feature in c for c in self.composers])

    def populate_vector_space(self, vocabulary, algorithm='ball_tree', build_tree=True):
        #todo the exact data structure used here will need optimisation
        """
        Input is like:
         DocumentFeature('1-GRAM', ('Seattle/N',)),
         DocumentFeature('1-GRAM', ('Senate/N',)),
         DocumentFeature('1-GRAM', ('September/N',)),
         DocumentFeature('1-GRAM', ('Service/N',)),
         DocumentFeature('AN', ('similar/J', 'agreement/N')),
         DocumentFeature('AN', ('tough/J', 'stance/N')),
         DocumentFeature('AN', ('year-ago/J', 'period/N'))
        """
        logging.debug('Populating vector space with algorithm %s and vocabulary %s', algorithm, vocabulary)
        logging.debug('Composer mapping is %s', self.composer_mapping)
        vectors = [c._get_vector(f)
                   for f in vocabulary
                   for c in self.composer_mapping[f.type]
                   if f.type in self.composer_mapping and f in c]
        if not vectors:
            raise ValueError('No vectors')
        self.feature_matrix = vstack(vectors)
        self.entry_index = [f for f in vocabulary for c in self.composer_mapping[f.type]
                            if f.type in self.composer_mapping and f in c]

        a, b = self.feature_matrix.shape
        if a < 50 and b < 50: # this may well be a test run, save some more debug info
            self.debug_entry_index = [(f, c) for f in vocabulary for c in self.composer_mapping[f.type]
                                      if f.type in self.composer_mapping and f in c]


        #self.entry_index = {i: ngram for i, ngram in enumerate(feature_list)}
        #assert len(feature_list) == self.feature_matrix.shape[0]
        #todo BallTree/KDTree only work with dense inputs

        if build_tree:
            logging.debug('Building BallTree for matrix of size %s', self.feature_matrix.shape)
            #self.nbrs = KDTree(n_neighbors=1, algorithm='kd_tree').fit(self.feature_matrix)
            self.nbrs = NearestNeighbors(metric=cosine, algorithm=algorithm, n_neighbors=2).fit(self.feature_matrix.A)
            logging.debug('Done building BallTree')
        return self.nbrs


    def write_vectors_to_disk(self, feature_types, vectors_path, new_entries_path, features_path):
        """
        Writes out the vectors, entries and features for all non-unigram features of this vector space to a
        Byblo-compatible file
        """
        logging.info('Writing all features to disk to %s', vectors_path)
        voc = self.composers[0].unigram_source.distrib_features_vocab
        m = self.feature_matrix
        entry_index = self.entry_index

        utils.write_vectors_to_disk(m, entry_index, voc, features_path, new_entries_path, vectors_path,
                                    entry_filter=lambda x: x.type in feature_types)
        logging.info('Done writing to disk')

    def _get_vector(self, feature):
        """
        Returns a set of vector for the specified ngram, one from each sub-source
        """
        return [(c.name, c._get_vector(feature).todense()) for c in self.composer_mapping[feature.type]]

    def _get_nearest_neighbours(self, feature):
        """
        Returns (composer, sim, neighbour) tuples for the given n-gram, one from each composer._get_vector
        Accepts structured features
        """
        res = []
        for comp_name, vector in self._get_vector(feature):
            distances, indices = self.nbrs.kneighbors(vector, return_distance=True)

            for dist, ind in zip(distances[0, :], indices[0, :]):
                similarity = 1 - dist
                neighbour = self.entry_index[ind]

                if (feature == neighbour and not self.include_self) or 1 - dist < self.sim_threshold:
                    continue
                else:
                    data = (comp_name, (neighbour, similarity))
                    res.append(data)
                    break
        return res

    def get_nearest_neighbours(self, feature):
        """
        Returns only the third element of what self._get_nearest_neighbours returns
        """
        #print feature, self._get_nearest_neighbours(feature)
        return map(itemgetter(1), self._get_nearest_neighbours(feature))

    def __str__(self):
        wrapped = ', '.join(str(c) for c in self.composers)
        return 'Composite[%s, mapping=%s]' % (wrapped, self.composer_mapping)


class PrecomputedSimilaritiesVectorSource(CompositeVectorSource):
    """
    Wraps a Byblo-computer Thesaurus in the interface of a CompositeVectorSource, deferring the get_nearest_neighbours
    method to the Thesaurus. This is different from UnigramDummyComposer as it contains a precomputed all-pairs
    similarity matrix and NOT vectors for each entry
    """
    feature_pattern = {'1-GRAM'}
    name = 'Precomputed'


    def __init__(self, thesaurus_files='', sim_threshold=0, include_self=False):
        '''
        :param thesaurus_files: List of **all-pairs similarities** files.
        :type thesaurus_files: list
        :param sim_threshold:
        :type sim_threshold: float
        :param include_self:
        :type include_self: bool
        '''
        self.th = Thesaurus(thesaurus_files=thesaurus_files, sim_threshold=sim_threshold, include_self=include_self)

    def _get_nearest_neighbours(self, feature):
    # Accepts structured features and strips the meta information from the feature and use as a string
    # Returns (composer, sim, neighbour) tuples
    # Feature structure is DocumentFeature('1-GRAM', ('Seattle/N',))

        # strip the structural info from feature for thes lookup
        res = self.th.get(feature.tokens_as_str())
        # put structural info back in
        return [(
                    'Byblo',
                    (# create a DocumentFeature object based on the string provided by thesaurus
                     DocumentFeature.from_string(x[0]),
                     x[1]
                    )
                )
                for x in res] if res else []

    def __contains__(self, feature):
        # strip the meta information from the feature and use as a string, thesaurus does not contain this info
        return '_'.join(map(str, feature.tokens)) in self.th

    def _get_vector(self, feature):
        raise ValueError('This is a precomputed neighbours object, it does not contain vectors.')

    def populate_vector_space(self, *args, **kwargs):
        #nothing to do, we have the all-pairs sim matrix already
        pass

    def __str__(self):
        return '[PrecomputedSimilaritiesVectorSource with %d entries]' % len(self.th)


class ConstantNeighbourVectorSource(VectorSource):
    """
    A thesaurus-like object which has
     1) a single neighbour for every possible entry
     2) a single random neighbour for every possible entry. That neighbour is chosen from the vocabulary that is
        passed in (as a dict {feature:index} )
    """
    name = 'Constant'

    def __init__(self, vocab=None):
        self.vocab = vocab


    def get_nearest_neighbours(self, feature):
        if self.vocab:
            v = choice(self.vocab.keys())
            return [(v, 1.0)]
        else:
            return [
                (
                    DocumentFeature('1-GRAM', (Token('b', 'N'),)),
                    1.0
                )
            ]

    def populate_vector_space(self, *args, **kwargs):
        pass


    def _get_vector(self):
        pass

    def __contains__(self, feature):
        return True
