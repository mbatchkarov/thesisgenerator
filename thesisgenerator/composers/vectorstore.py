from abc import ABCMeta, abstractmethod
from collections import defaultdict
from itertools import chain, groupby
import logging
from random import choice

from operator import itemgetter
from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors
import numpy as np
import scipy.sparse as sp
from scipy.sparse import vstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.random_projection import SparseRandomProjection

from thesisgenerator.plugins.thesaurus_loader import Thesaurus
from thesisgenerator.plugins.tokenizers import DocumentFeature, Token


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

        v = DictVectorizer(sparse=True, dtype=np.int32)

        # distributional features of each unigram in the loaded file
        self.feature_matrix = v.fit_transform([dict(fv) for fv in thesaurus.itervalues()])

        # Token -> row number in self.feature_matrix that holds corresponding vector
        self.entry_index = {Token(*fv.split('/')): i for (i, fv) in enumerate(thesaurus.keys())}

        # the set of all distributional features, for unit testing only
        self.distrib_features_vocab = v.vocabulary_

        self.available_pos = set(t.pos for t in self.entry_index.keys())
        if reduce_dimensionality:
            logging.info('Reducing dimensionality of unigram vectors from %s to %s',
                         self.feature_matrix.shape[1], dimensions)
            self.transformer = SparseRandomProjection(n_components=dimensions)
            self.feature_matrix = self.transformer.fit_transform(self.feature_matrix)
            self.distrib_features_vocab = None


    def _get_vector(self, tokens):
        # word must be an iterable of Token objects
        try:
            row = self.entry_index[tokens[0]]
            if len(tokens) > 1:
                logging.warn('Attempting to get unigram vector of n-gram %r', tokens)
        except KeyError:
            return None
        return self.feature_matrix[row, :]

    def __contains__(self, feature):
        """
        Accept all unigrams that we have a vector for
        the thing is a unigram and we have a corpus-based vector for that unigram
        """
        return feature.type in self.feature_pattern and feature.tokens[0] in self.entry_index

    def __str__(self):
        return '[UnigramVectorSource with %d %d-dimensional entries]' % self.feature_matrix.shape

    def __len__(self):
        return len(self.entry_index)


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

    def _get_vector(self, tokens):
        return self.unigram_source._get_vector(tokens)

    def __str__(self):
        return '[UnigramDummyComposer wrapping %s]' % self.unigram_source


class OxfordSvoComposer(Composer):
    name = 'DummySVO'
    feature_pattern = {'SVO'}

    def __init__(self, unigram_source=None):
        super(OxfordSvoComposer, self).__init__(unigram_source)
        if 'V' not in unigram_source.available_pos or \
                        'N' not in unigram_source.available_pos:
            raise ValueError('This composer requires a noun and verb unigram vector sources')

    def __contains__(self, feature):
        """
        Accept all subject-verb-object phrases where we have a corpus-observed vector for each unigram
        """
        if feature.type not in self.feature_pattern:
            # ignore non-SVO features
            return False

        for token in feature.tokens:
            if DocumentFeature('1-GRAM', (token,)) not in self.unigram_source:
                # ignore ANs containing unknown nouns
                return False

        return True

    def _get_vector(self, tokens):
        #todo currently returns just the verb vector, which is wrong
        return self.unigram_source._get_vector((tokens[1],))


class AdditiveComposer(Composer):
    name = 'Add'
    # composers in general work with n-grams (for simplicity n<4)
    feature_pattern = {'2-GRAM', '3-GRAM'}

    def __init__(self, unigram_source=None):
        super(AdditiveComposer, self).__init__(unigram_source)

    def _get_vector(self, tokens):
        return sum(self.unigram_source._get_vector((token,)) for token in tokens)

    def __contains__(self, feature):
        """
        Contains all sequences of words where we have a distrib vector for each unigram
        they contain. Rejects unigrams.
        """
        if feature.type == '1-GRAM' or feature.type not in self.feature_pattern:
            # no point in composing single-word document features
            return False

        acceptable = True
        for unigram in feature.tokens:
            if DocumentFeature('1-GRAM', (unigram,)) not in self.unigram_source:
                # ignore n-grams containing unknown unigrams
                acceptable = False
                break
        return acceptable

    def __str__(self):
        return '[AdditiveComposer with %d unigram entries]' % (len(self.unigram_source))


class MultiplicativeComposer(AdditiveComposer):
    name = 'Mult'

    def __init__(self, unigram_source=None):
        super(MultiplicativeComposer, self).__init__(unigram_source)

    def _get_vector(self, tokens):
        return reduce(sp.csr_matrix.multiply,
                      [self.unigram_source._get_vector((t,)) for t in tokens[1:]],
                      self.unigram_source._get_vector([tokens[0]]))

    def __str__(self):
        return '[MultiplicativeComposer with %d unigram entries]' % (len(self.unigram_source))


class BaroniComposer(Composer):
    # BaroniComposer composes AN features
    feature_pattern = {'AN'}
    name = 'Baroni'

    def __init__(self, unigram_source=None):
        super(BaroniComposer, self).__init__(unigram_source)
        if 'N' not in unigram_source.available_pos:
            raise ValueError('This composer requires a noun unigram vector source')

    def __contains__(self, feature):
        """
        Accept all adjective-noun phrases where we have a corpus-observed vector for the noun and
        a learnt matrix (through PLSR) for the adjective
        """
        if feature.type not in self.feature_pattern:
            # ignore non-AN features
            return False

        adj, noun = feature.tokens
        if DocumentFeature('1-GRAM', (noun,)) not in self.unigram_source:
            # ignore ANs containing unknown nouns
            return False

        # todo enable this
        #if adj not in self.adjective_matrices.keys():
        #        # ignore ANs containing unknown adjectives
        #        continue

        return True

    def _get_vector(self, tokens):
        #todo currently returns just the noun vector, which is wrong
        return self.unigram_source._get_vector((tokens[-1], ))


class CompositeVectorSource(VectorSource):
    def __init__(self, composers, sim_threshold, include_self):
        self.composers = composers
        self.sim_threshold = sim_threshold
        self.include_self = include_self

        self.nbrs, self.feature_matrix, entry_index = [None] * 3     # computed by self.build_peripheral_space()
        self.composers = composers
        self.composer_mapping = defaultdict(set) # feature type -> {composer object}
        #tmp = OrderedDict()
        for c in self.composers:
            for p in c.feature_pattern:
                self.composer_mapping[p].add(c)
                #self.composer_mapping.update(tmp)

    def __contains__(self, feature):
        return any(feature in c for c in self.composers)

    def populate_vector_space(self, vocabulary, algorithm='ball_tree', build_tree=True):
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
        logging.debug('Populating vector space with algorithm %s and vocabulary %s', algorithm, vocabulary)
        logging.debug('Composer mapping is %s', self.composer_mapping)
        vectors = [c._get_vector(f.tokens)
                   for f in vocabulary
                   for c in self.composer_mapping[f.type]
                   if f.type in self.composer_mapping and f in c]

        self.feature_matrix = vstack(vectors)
        feature_list = [f for f in vocabulary for _ in self.composer_mapping[f.type]]
        #todo test if this entry index is correct
        self.entry_index = {i: ngram for i, ngram in enumerate(feature_list)}
        #assert len(feature_list) == self.feature_matrix.shape[0]
        #todo BallTree/KDTree only work with dense inputs

        if build_tree:
            logging.debug('Building BallTree for matrix of size %s', self.feature_matrix.shape)
            #self.nbrs = KDTree(n_neighbors=1, algorithm='kd_tree').fit(self.feature_matrix)
            self.nbrs = NearestNeighbors(metric=cosine, algorithm=algorithm, n_neighbors=2).fit(self.feature_matrix.A)
            logging.debug('Done building BallTree')
        return self.nbrs

    def write_vectors_to_disk(self, vectors_path, new_entries_path, features_path, feature_type):
        """
        Writes out the vectors, entries and features for all non-unigram features of this vector space to a
        Byblo-compatible file
        """
        logging.info('Writing all features to disk to %s', vectors_path)
        voc = self.composers[0].unigram_source.distrib_features_vocab

        new_byblo_entries = {}
        sorted_voc = np.array([x[0] for x in sorted(voc.iteritems(), key=itemgetter(1))])
        m = self.feature_matrix
        things = zip(m.row, m.col, m.data)
        selected_rows = []
        with open(vectors_path, 'wb') as outfile:
            for row, group in groupby(things, lambda x: x[0]):
                feature = self.entry_index[row]
                if feature.type == feature_type:
                    selected_rows.append(row)
                    ngrams_and_counts = [(sorted_voc[x[1]], x[2]) for x in group]
                    #logging.info(feature)
                    outfile.write('%s\t%s\n' % (
                        feature.tokens_as_str(),
                        '\t'.join(map(str, chain.from_iterable(ngrams_and_counts)))
                    ))
                    new_byblo_entries[feature] = sum(x[1] for x in ngrams_and_counts)
                if row % 100 == 0:
                    logging.info('Processed %d vectors', row)

        with open(new_entries_path, 'w') as outfile:
            for entry, count in new_byblo_entries.iteritems():
                outfile.write('%s\t%d\n' % (entry.tokens_as_str(), count))

        with open(features_path, 'w') as outfile:
            if selected_rows: # guard agains empty files
                feature_sums = np.array(m.tocsr()[selected_rows].sum(axis=0))[0, :]
                for feature, count in zip(sorted_voc, feature_sums):
                    if count > 0:
                        outfile.write('%s\t%d\n' % (feature, count))
        logging.info('Done writing to disk')

    def _get_vector(self, feature):
        """
        Returns a set of vector for the specified ngram, one from each sub-source
        """
        feature_type, tokens = feature.type, feature.tokens
        return [(c.name, c._get_vector(tokens).todense()) for c in self.composer_mapping[feature_type]]

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

    def get_nearest_neighbours(self, ngram):
        """
        Returns only the third element of what self._get_nearest_neighbours returns
        """
        #print ngram, self._get_nearest_neighbours(ngram)
        return map(itemgetter(1), self._get_nearest_neighbours(ngram))

    def __str__(self):
        wrapped = ', '.join(str(c) for c in self.composers)
        return 'Composite[%s, mapping=%s]' % (wrapped, self.composer_mapping)

    def __repr__(self):
        return self.__str__()


class PrecomputedSimilaritiesVectorSource(CompositeVectorSource):
    """
    Wraps a Byblo-computer Thesaurus in the interface of a CompositeVectorSource, deferring the get_nearest_neighbours
    method to the Thesaurus. Only handles features of the form ('1-GRAM', (X,))
    """
    feature_pattern = {'1-GRAM'}
    name = 'BybloThes'

    def __init__(self, thesaurus_files='', sim_threshold=0, include_self=False):
        self.th = Thesaurus(thesaurus_files=thesaurus_files, sim_threshold=sim_threshold, include_self=include_self)

    def _get_nearest_neighbours(self, feature):
    # Accepts structured features and strips the meta information from the feature and use as a string
    # Returns (composer, sim, neighbour) tuples
    # Feature structure is ('1-GRAM', ('Seattle/N',))

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
        return ' '.join(map(str, feature.tokens)) in self.th

    def keys(self):
        # todo this needs to be removed from the interface of this class
        return self.th.keys()

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