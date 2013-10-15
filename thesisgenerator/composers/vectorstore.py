from abc import ABCMeta, abstractmethod
from collections import defaultdict, OrderedDict
from random import choice

from operator import itemgetter
from scipy.spatial.distance import cosine
from sklearn.neighbors import BallTree
import numpy
import scipy.sparse as sp
from numpy import vstack
from sklearn.feature_extraction import DictVectorizer

from thesisgenerator.plugins.thesaurus_loader import Thesaurus


class VectorSource(object):
    __metaclass__ = ABCMeta

    feature_pattern = {}    # each VectorSource can work with a set of feature types

    @abstractmethod
    def __contains__(self, features):
        """
        Filters out document features that cannot be handled by the implementing model. For instance,
        BaroniComposer cannot handle noun compounds or AN compounds for some adjectives. Features
        are assumed to be generated externally
        """
        pass

    @abstractmethod
    def _get_vector(self, word):
        pass


#class ExactMatchVectorSource(VectorSource):
#    feature_pattern = {'1-GRAM', '2-GRAM', '3-GRAM'}
#    name = 'Exact'
#
#    def __contains__(self, features):
#        return True
#
#    def _get_vector(self, word):
#        raise ValueError('This class cannot provide vectors')


class UnigramVectorSource(VectorSource):
    feature_pattern = {'1-GRAM'}
    name = 'Lex'

    def __init__(self, unigram_paths):
        if not unigram_paths:
            raise ValueError('You must provide a unigram vector file')

        thesaurus = Thesaurus(
            thesaurus_files=unigram_paths,
            sim_threshold=0,
            include_self=False)

        v = DictVectorizer(sparse=True, dtype=numpy.int32)

        # distributional features of each unigram in the loaded file
        self.feature_matrix = v.fit_transform([dict(fv) for fv in thesaurus.itervalues()])

        # unigram -> row number in self.feature_matrix that holds corresponding vector
        self.entry_index = {fv: i for (i, fv) in enumerate(thesaurus.keys())}

        # the set of all distributional features, for unit testing only
        self.distrib_features_vocab = v.vocabulary_

    def _get_vector(self, word):
        # word must be a a string or an iterable. If it's the latter, the first item is used

        if hasattr(word, '__iter__'):  # False for strings, true for lists/tuples
            if len(word) == 1:
                word = word[0]
            else:
                raise ValueError('Attempting to get unigram vector of non-unigram {}'.format(word))

        try:
            row = self.entry_index[word]
        except KeyError:
            return None
        return self.feature_matrix[row, :]

    #def compose(self, words):
    #    """
    #    Takes a sequence of words and returns a vector for them. This implementation does
    #    not bother with composition, just returns a unigram vector. The thing passed in must
    #    be a one-element iterable. Subclasses override to implement composition
    #    """
    #    return self.get_vector(words[0])

    def __contains__(self, feature):
        """
        Accept all unigrams that we have a vector for
        the thing is a unigram and we have a corpus-based vector for that unigram
        """
        return feature[0] in self.feature_pattern and feature[1][0] in self.entry_index.keys()


class Composer(VectorSource):
    def __init__(self, unigram_source=None):
        if not unigram_source:
            raise ValueError('Composers need a unigram vector source')
        self.unigram_source = unigram_source


class UnigramDummyComposer(Composer):
    name = 'Lexical'
    feature_pattern = {'1-GRAM'}

    def __init__(self, unigram_source=None):
        super(UnigramDummyComposer, self).__init__(unigram_source)

    def __contains__(self, feature):
        return feature in self.unigram_source

    def _get_vector(self, sequence):
        return self.unigram_source._get_vector(sequence)


class AdditiveComposer(Composer):
    name = 'Add'
    # composers in general work with n-grams (for simplicity n<4)
    feature_pattern = {'2-GRAM', '3-GRAM'}

    def __init__(self, unigram_source=None):
        super(AdditiveComposer, self).__init__(unigram_source)

    def _get_vector(self, sequence):
        return sum(self.unigram_source._get_vector(word) for word in sequence)

    def __contains__(self, f):
        """
        Contains all sequences of words where we have a distrib vector for each unigram
        they contain. Rejects unigrams.
        """
        if f[0] == '1-GRAM':
            # no point in composing single-word document features
            return False

        acceptable = True
        for unigram in f[1]:
            if ('1-GRAM', (unigram,)) not in self.unigram_source:
                # ignore n-grams containing unknown unigrams
                acceptable = False
                break
        return acceptable


class MultiplicativeComposer(AdditiveComposer):
    name = 'Mult'

    def __init__(self, unigram_source=None):
        super(MultiplicativeComposer, self).__init__(unigram_source)

    def _get_vector(self, sequence):
        return reduce(sp.csr_matrix.multiply,
                      map(self.unigram_source._get_vector, sequence[1:]),
                      self.unigram_source._get_vector(sequence[0]))


class BaroniComposer(Composer):
    # BaroniComposer composes AN features
    feature_pattern = {'AN'}
    name = 'Baroni'

    def __init__(self, unigram_source=None):
        super(BaroniComposer, self).__init__(unigram_source)

    def __contains__(self, feature):
        """
        Accept all adjective-noun phrases where we have a corpus-observed vector for the noun and
        a learnt matrix (through PLSR) for the adjective
        """
        if feature[0] not in self.feature_pattern:
            # ignore non-AN features
            return False

        adj, noun = feature[1]
        if noun not in self.unigram_source.entry_index.keys():
            # ignore ANs containing unknown nouns
            return False

        # todo enable this
        #if adj not in self.adjective_matrices.keys():
        #        # ignore ANs containing unknown adjectives
        #        continue

        return True

    def _get_vector(self, sequence):
        #todo currently returns just the noun vector, which is wrong
        return self.unigram_source._get_vector(sequence[-1])


class CompositeVectorSource(VectorSource):
    def __init__(self, unigram_source, composers, sim_threshold, include_self):
        self.unigram_source = unigram_source
        self.composers = composers
        self.sim_threshold = sim_threshold
        self.include_self = include_self

        self.nbrs, self.feature_matrix, entry_index = [None] * 3     # computed by self.build_peripheral_space()

        #if composers:
        #    self.composers.append(self.unigram_source)
        self.composers = composers
        self.composer_mapping = OrderedDict()
        tmp = defaultdict(set) # feature type -> {composer object}
        for c in self.composers:
            for p in c.feature_pattern:
                tmp[p].add(c)
        self.composer_mapping.update(tmp)

    def __contains__(self, item):
    #for c in self.composers:
    #print c
    #print c.accept_features(features)
    #print
    #    return {f for c in self.composers for f in c.__contains__(features)}
        return any(item in c for c in self.composers)

    def populate_vector_space(self, vocabulary):
        #todo the exact data structure used here will need optimisation
        """
        Input is like:
         ('1-GRAM', ('Seattle/N',)),
         ('1-GRAM', ('Senate/N',)),
         ('1-GRAM', ('September/N',)),
         ('1-GRAM', ('Service/N',)),
         ('AN', ('similar/J', 'agreement/N')),
         ('AN', ('tough/J', 'stance/N')),
         ('AN', ('year-ago/J', 'period/N'))
        """

        vectors = [c._get_vector(data).A
                   for (feature_type, data) in vocabulary
                   for c in self.composer_mapping[feature_type]
                   if feature_type in self.composer_mapping and (feature_type, data) in c]
        self.feature_matrix = vstack(vectors)

        feature_list = [ngram for ngram in vocabulary for _ in self.composer_mapping[ngram[0]]]
        #todo test if this entry index is correct
        self.entry_index = {i: ngram for i, ngram in enumerate(feature_list)}
        #assert len(feature_list) == self.feature_matrix.shape[0]
        #todo BallTree/KDTree only work with dense inputs
        #self.nbrs = KDTree(n_neighbors=1, algorithm='kd_tree').fit(self.feature_matrix)
        self.nbrs = BallTree(self.feature_matrix, metric=cosine)

    def _get_vector(self, ngram):
        """
        Returns a set of vector for the specified ngram, one from each sub-source
        """
        feature_type, data = ngram
        return [(c.name, c._get_vector(data).todense()) for c in self.composer_mapping[feature_type]]

    def _get_nearest_neighbours(self, ngram):
        """
        Returns (composer, sim, neighbour) tuples for the given n-gram, one from each composer._get_vector
        Accepts structured features
        """
        res = []
        #print 'Composer\t\t\tdist\t\t\tneighbour'
        for comp_name, vector in self._get_vector(ngram):
            #dist, ind = self.nbrs.kneighbors(vector)
            dist, ind = self.nbrs.query(vector, k=1, return_distance=True)
            data = (comp_name, (dist[0][0], self.entry_index[ind[0][0]]))
            #print '{}\t\t\t{}\t\t\t{}'.format(*data)
            #todo tests for this if and the one below
            if ngram == data[1][1] and not self.include_self:
                continue
            if 1 - dist < self.sim_threshold:
                continue
            res.append(data)
        return res

    def get_nearest_neighbours(self, ngram):
        """
        Returns only the third element of what self._get_nearest_neighbours returns
        """
        #print ngram, self._get_nearest_neighbours(ngram)
        return map(itemgetter(1), self._get_nearest_neighbours(ngram))


class PrecomputedSimilaritiesVectorSource(CompositeVectorSource):
    """
    Wraps a Byblo-computer Thesaurus in the interface of a CompositeVectorSource, deferring the get_nearest_neighbours
    method to the Thesaurus. Only handles features of the form ('1-GRAM', (X,))
    """
    feature_pattern = {'1-GRAM'}
    name = 'BybloThes'

    def __init__(self, thesaurus_files='', sim_threshold=0, include_self=False):
        self.th = Thesaurus(thesaurus_files=thesaurus_files, sim_threshold=sim_threshold, include_self=include_self)

    def _get_nearest_neighbours(self, word):
    # Accepts structured features and strips the meta information from the feature and use as a string
    # Returns (composer, sim, neighbour) tuples
    # Feature structure is ('1-GRAM', ('Seattle/N',))

        # strip the structural info from feature for thes lookup
        res = self.th.get(word[1][0])
        # put structural info back in
        return [(
                    'Byblo',
                    (
                        ('1-GRAM', (x[0],)),
                        x[1]
                    )
                )
                for x in res] if res else []

    def __contains__(self, features):
        # strip the meta information from the feature and use as a string, thesaurus does not contain this info

        return [x for x in features if x[1][0] in self.th.keys()]

    def keys(self):
        # todo this needs to be removed from the interface of this class
        return self.th.keys()

    def __contains__(self, item):
        #Accepts structured features
        return item[1][0] in self.th

    def populate_vector_space(self, vocabulary):
        #nothing to do, we have the all-pairs sim matrix already
        pass


class ConstantNeighbourVectorSource(VectorSource):
    """
    A thesaurus-like object which has
     1) a single neighbour for every possible entry
     2) a single random neighbour for every possible entry. That neighbour is chosen from the vocabulary that is
        passed in (as a dict {feature:index} )
    """

    def __init__(self, vocab=None):
        self.vocab = vocab


    def get_nearest_neighbours(self, thing):
        if self.vocab:
            v = choice(self.vocab.keys())
            return [(v, 1.0)]
        else:
            return [
                (
                    ('1-GRAM', ('b/n',)),
                    1.0
                )
            ]

    def populate_vector_space(self, thing):
        pass

    def __contains__(self):
        pass

    def _get_vector(self):
        pass

    def __contains__(self, item):
        return True