from collections import defaultdict
import inspect
import numpy
import scipy.sparse as sp
from sklearn.feature_extraction import DictVectorizer
from thesisgenerator.plugins.thesaurus_loader import Thesaurus
from thesisgenerator.utils.data_utils import get_named_object


class FeatureAcceptor(object):
    feature_pattern = {} # each FeatureAcceptor can work with a set of feature types

    def accept_features(self, features):
        """
        Filters out document features that cannot be handled by the implementing model. For instance,
        BaroniComposer cannot handle noun compounds or AN compounds for some adjectives. Features
        are assumed to be generated externally
        """
        raise NotImplementedError('Subclasses must override')


class BasicVectorSource(FeatureAcceptor):
    #todo change this to a dict-like object
    feature_pattern = {'1-GRAM'}

    def __init__(self, files=None):
        if not files:
            raise ValueError('You must provide a unigram vector file')

        thesaurus = Thesaurus(
            thesaurus_files=files,
            sim_threshold=0,
            include_self=False)

        v = DictVectorizer(sparse=True, dtype=numpy.int32)
        self.feature_matrix = v.fit_transform([dict(fv) for fv in thesaurus.itervalues()])
        self.entry_index = {fv: i for (i, fv) in enumerate(thesaurus.keys())}
        self.distrib_features_vocab = v.vocabulary_

    def get_vector(self, word):
        # word must be a a string
        try:
            row = self.entry_index[word]
        except KeyError:
            return None
        return self.feature_matrix[row, :]

    def compose(self, words):
        """
        Takes a sequence of words and returns a vector for them. This implementation does
        not bother with composition, just returns a unigram vector. The thing passed in must
        be a one-element iterable. Subclasses override to implement composition
        """
        return self.get_vector(words)

    def accept_features(self, features):
        """
        Accept all unigrams that we have a vector for
        """
        return {t for t in features
                if t[0] in self.feature_pattern and # the thing is a unigram
                   t[1][0] in self.entry_index.keys() # we have a corpus-based vector for that unigram
        }


class Composer(BasicVectorSource):
    def __init__(self, unigram_source=None):
        if not unigram_source:
            raise ValueError('Composers need a unigram vector source')
        self.unigram_source = unigram_source

    def get_vector(self, word):
        # delegate the work to the object that can do it
        return self.unigram_source.get_vector()

    def compose(self, words):
        # prevent subclasses of Composer of falling back to the implementation of BasicVectorSource
        raise NotImplementedError('Subclasses must override')


class AdditiveComposer(Composer):
    # composers in general work with n-grams (for simplicity n<4)
    feature_pattern = {'2-GRAM', '3-GRAM'}

    def __init__(self, unigram_source=None):
        super(AdditiveComposer, self).__init__(unigram_source)

    def compose(self, sequence):
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
    def __init__(self, unigram_source=None):
        super(MultiplicativeComposer, self).__init__(unigram_source)

    def compose(self, sequence):
        return reduce(sp.csr_matrix.multiply,
                      map(self.unigram_source.get_vector, sequence[1:]),
                      self.unigram_source.get_vector(sequence[0]))


class BaroniComposer(Composer):
    # BaroniComposer composes AN features
    feature_pattern = {'AN'}

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

    def accept_features(self, features):
        """
        Accept all sequences of words where we have a distrib vector for each unigram
        they contain. Rejects unigrams. This implementation is suitable for AdditiveComposer and
        MultiplicativeComposer.
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


class CompositeVectorSource(FeatureAcceptor):
    def __init__(self, conf):
        self.unigram_source = BasicVectorSource(conf['unigram_paths'])
        self.composers = []

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

        self.composer_mapping = defaultdict(set) # feature type -> {composer object}
        for c in self.composers:
            for p in c.feature_pattern:
                self.composer_mapping[p].add(c)

    def accept_features(self, features):
    #for c in self.composers:
    #print c
    #print c.accept_features(features)
    #print
        return {f for c in self.composers for f in c.accept_features(features)}

    def build_peripheral_space(self, vocabulary):
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
        # todo build a matrix first, then place in KD-tree
        for (feature_type, data) in vocabulary:
            for c in self.composer_mapping[feature_type]:
                v = c.compose(data)
                #print data
                #print v.shape

                #self.nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(X)

    def get_vectors(self, ngram):
        """
        Returns a set of vector for the specified ngram, one from each sub-source
        """
        #todo implement me
        raise NotImplementedError('todo')