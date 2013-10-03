from abc import ABCMeta, abstractmethod
from collections import defaultdict, OrderedDict
import inspect
import numpy
import scipy.sparse as sp
from scipy.sparse import vstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import NearestNeighbors
from thesisgenerator.plugins.thesaurus_loader import Thesaurus
from thesisgenerator.utils.data_utils import get_named_object


class VectorSource(object):
    __metaclass__ = ABCMeta

    feature_pattern = {} # each VectorSource can work with a set of feature types

    @abstractmethod
    def accept_features(self, features):
        """
        Filters out document features that cannot be handled by the implementing model. For instance,
        BaroniComposer cannot handle noun compounds or AN compounds for some adjectives. Features
        are assumed to be generated externally
        """
        pass

    @abstractmethod
    def get_vector(self, word):
        pass


class UnigramVectorSource(VectorSource):
    #todo change this to a dict-like object
    feature_pattern = {'1-GRAM'}
    name = 'Lex'

    def __init__(self, files=None):
        if not files:
            raise ValueError('You must provide a unigram vector file')

        thesaurus = Thesaurus(
            thesaurus_files=files,
            sim_threshold=0,
            include_self=False)

        v = DictVectorizer(sparse=True, dtype=numpy.int32)

        # distributional features of each unigram in the loaded file
        self.feature_matrix = v.fit_transform([dict(fv) for fv in thesaurus.itervalues()])

        # unigram -> row number in self.feature_matrix that holds corresponding vector
        self.entry_index = {fv: i for (i, fv) in enumerate(thesaurus.keys())}

        # the set of all distributional features, for unit testing only
        self.distrib_features_vocab = v.vocabulary_

    def get_vector(self, word):
        # word must be a a string
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

    def accept_features(self, features):
        """
        Accept all unigrams that we have a vector for
        """
        return {t for t in features
                if t[0] in self.feature_pattern and # the thing is a unigram
                   t[1][0] in self.entry_index.keys() # we have a corpus-based vector for that unigram
        }


class Composer(VectorSource):
    def __init__(self, unigram_source=None):
        if not unigram_source:
            raise ValueError('Composers need a unigram vector source')
        self.unigram_source = unigram_source


class AdditiveComposer(Composer):
    name = 'Add'
    # composers in general work with n-grams (for simplicity n<4)
    feature_pattern = {'2-GRAM', '3-GRAM'}

    def __init__(self, unigram_source=None):
        super(AdditiveComposer, self).__init__(unigram_source)

    def get_vector(self, sequence):
        return sum(self.unigram_source.get_vector(word) for word in sequence)

    def accept_features(self, features):
        """
        Accept all sequences of words where we have a distrib vector for each unigram
        they contain. Rejects unigrams.
        """
        accepted_features = set()
        for f in features:
            acceptable = True
            if f[0] == '1-GRAM':
                # no point in composing single-word document features
                continue

            for unigram in f[1]:
                if unigram not in self.unigram_source.entry_index.keys():
                    # ignore n-grams containing unknown unigrams
                    acceptable = False
                    break
            if acceptable:
                accepted_features.add(f)

        return accepted_features


class MultiplicativeComposer(AdditiveComposer):
    name = 'Mult'

    def __init__(self, unigram_source=None):
        super(MultiplicativeComposer, self).__init__(unigram_source)

    def get_vector(self, sequence):
        return reduce(sp.csr_matrix.multiply,
                      map(self.unigram_source.get_vector, sequence[1:]),
                      self.unigram_source.get_vector(sequence[0]))


class BaroniComposer(Composer):
    # BaroniComposer composes AN features
    feature_pattern = {'AN'}
    name = 'Baroni'

    def __init__(self, unigram_source=None):
        super(BaroniComposer, self).__init__(unigram_source)

    def accept_features(self, features):
        """
        Accept all adjective-noun phrases where we have a corpus-observed vector for the noun and
        a learnt matrix (through PLSR) for the adjective
        """
        accepted_features = set()
        for f in features:
            if f[0] not in self.feature_pattern:
                # ignore non-AN features
                continue
            adj, noun = f[1]
            if noun not in self.unigram_source.entry_index.keys():
                # ignore ANs containing unknown nouns
                continue

            # todo enable this
            #if adj not in self.adjective_matrices.keys():
            #        # ignore ANs containing unknown adjectives
            #        continue

            accepted_features.add(f)
        return accepted_features

    def get_vector(self, sequence):
        #todo currently returns just the noun vector, which is wrong
        return self.unigram_source.get_vector(sequence[-1])


class CompositeVectorSource(VectorSource):
    def __init__(self, conf):
        self.unigram_source = UnigramVectorSource(conf['unigram_paths'])
        self.composers = []
        self.nbrs, self.feature_matrix, entry_index = [None] * 3 # computed by self.build_peripheral_space()

        if conf['include_unigram_features']:
            self.composers.append(self.unigram_source)
        for section in conf:
            if 'composer' in section and conf[section]['run']:
                composer_class = get_named_object(section)
                # todo the object must only take keyword arguments
                initialize_args = inspect.getargspec(composer_class.__init__)[0]
                opts = conf[section]
                args = {arg: val for arg, val in opts.items() if arg in initialize_args}
                args['unigram_source'] = self.unigram_source
                self.composers.append(composer_class(**args))

        self.composer_mapping = OrderedDict()
        tmp = defaultdict(set) # feature type -> {composer object}
        for c in self.composers:
            for p in c.feature_pattern:
                tmp[p].add(c)
        self.composer_mapping.update(tmp)

    def accept_features(self, features):
    #for c in self.composers:
    #print c
    #print c.accept_features(features)
    #print
        return {f for c in self.composers for f in c.accept_features(features)}

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

        self.feature_matrix = vstack(c.get_vector(data).tolil()
                                     for (feature_type, data) in vocabulary
                                     for c in self.composer_mapping[feature_type]).tocsr()

        feature_list = [ngram for ngram in vocabulary for c in self.composer_mapping[ngram[0]]]
        #todo test if this entry index is correct
        self.entry_index = {i: ngram for i, ngram in enumerate(feature_list)}
        assert len(feature_list) == self.feature_matrix.shape[0]
        self.nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(self.feature_matrix)

    def get_vector(self, ngram):
        """
        Returns a set of vector for the specified ngram, one from each sub-source
        """
        feature_type, data = ngram
        return [(c.name, c.get_vector(data).todense()) for c in self.composer_mapping[feature_type]]

    def get_nearest_neighbours(self, ngram):
        """
        Composes
        """
        print 'Composer\t\t\tsim\t\t\tneighbour'
        for composer, vector in self.get_vector(ngram):
            dist, ind = self.nbrs.kneighbors(vector)
            print '{}\t\t\t{}\t\t\t{}'.format(composer, 1 - dist[0][0], self.entry_index[ind[0][0]])
            #return self.nbrs.