import logging
from random import choice
from pickle import load

import numpy as np
import scipy.sparse as sp
from discoutils.thesaurus_loader import Thesaurus, Vectors

from discoutils.tokens import DocumentFeature




def check_vectors(unigram_source):
    if not unigram_source:
        raise ValueError('Composers need a unigram vector source')
    if not hasattr(unigram_source, 'get_vector'):
        raise ValueError('Creating a composer requires a Vectors data structure that holds unigram vectors')
    return unigram_source


class ComposerMixin(object):
    def compose_all(self, things):
        """
        Returns all unigrams and all composed `things` as a matrix, like
        :param things:
        :type things:
        :return:
        :rtype:
        """
        # todo return mat, cols, rows; append them to  self.mat, rows
        pass


class AdditiveComposer(Vectors, ComposerMixin):
    name = 'Add'
    # composers in general work with n-grams (for simplicity n<4)
    entry_types = {'2-GRAM', '3-GRAM', 'AN', 'NN', 'VO', 'SVO'}

    def __init__(self, unigram_source):
        self.unigram_source = check_vectors(unigram_source)
        self.function = np.add

    def get_vector(self, feature):
        """
        :type feature: DocumentFeature
        :rtype: scipy.sparse.csr_matrix
        """
        if isinstance(feature, str):
            feature = DocumentFeature.from_string(feature)
        return sp.csr_matrix(reduce(self.function,
                                    [self.unigram_source.get_vector(t.tokens_as_str()).A for t in feature[:]]))


    def contains_impl(self, feature):
        """
        Contains all sequences of words where we have a distrib vector for each unigram
        they contain. Rejects unigrams.
        """
        if isinstance(feature, str):
            feature = DocumentFeature.from_string(feature)

        if feature.type not in self.entry_types:
            # no point in trying
            return False
        return all(f.tokens_as_str() in self.unigram_source for f in feature[:])

    def __contains__(self, feature):
        return self.contains_impl(feature)

    def __str__(self):
        return '[%s with %d unigram entries]' % (self.__class__.__name__, len(self.unigram_source))


class MultiplicativeComposer(AdditiveComposer):
    name = 'Mult'

    def __init__(self, unigram_source):
        self.unigram_source = check_vectors(unigram_source)
        self.function = np.multiply

    def __contains__(self, feature):
        return self.contains_impl(feature)


class MinComposer(MultiplicativeComposer):
    name = 'Min'

    def __init__(self, unigram_source):
        self.unigram_source = check_vectors(unigram_source)
        self.function = lambda m, n: np.minimum(m, n)

    def __contains__(self, feature):
        return self.contains_impl(feature)


class MaxComposer(MinComposer):
    name = 'Max'

    def __init__(self, unigram_source):
        self.unigram_source = check_vectors(unigram_source)
        self.function = lambda m, n: np.maximum(m, n)

    def __contains__(self, feature):
        return self.contains_impl(feature)


class LeftmostWordComposer(AdditiveComposer):
    name = 'Left'

    def __init__(self, unigram_source):
        self.unigram_source = check_vectors(unigram_source)
        self.hardcoded_index = 0
        self.entry_types = {'2-GRAM', '3-GRAM', 'AN', 'NN', 'VO', 'SVO'}


    def get_vector(self, feature):
        if isinstance(feature, str):
            feature = DocumentFeature.from_string(feature)
        return self.unigram_source.get_vector(feature[self.hardcoded_index].tokens_as_str())

    def contains_impl(self, feature):
        if isinstance(feature, str):
            feature = DocumentFeature.from_string(feature)
        if feature.type not in self.entry_types:
            # no point in composing single-word document features
            return False
        return feature[self.hardcoded_index].tokens_as_str() in self.unigram_source

    def __contains__(self, feature):
        return self.contains_impl(feature)


class RightmostWordComposer(LeftmostWordComposer):
    name = 'Right'

    def __init__(self, unigram_source):
        self.unigram_source = check_vectors(unigram_source)
        self.hardcoded_index = -1


class BaroniComposer(Vectors, ComposerMixin):
    # BaroniComposer composes AN features
    entry_types = {'AN', 'NN'}
    name = 'Baroni'

    def __init__(self, unigram_source, pretrained_model_file):
        self.unigram_source = check_vectors(unigram_source)
        if not pretrained_model_file:
            logging.error('Expected filename, got %s', pretrained_model_file)
            raise ValueError('Model file required to perform composition.')
        with open(pretrained_model_file) as infile:
            self._composer = load(infile)

        # verify the composer's internal structure matches the unigram source
        self.available_modifiers = set(self._composer.function_space.id2row)
        features = self._composer.composed_id2column
        dimensionality = len(self.composer.composed_id2column)

        assert unigram_source.distrib_features_vocab == self._composer.composed_id2column
        self.dissect_core_space = unigram_source.dissect_core_space

        # check composed space's columns matche core space's (=unigram source)'s columns
        assert self.dissect_core_space.id2column == self.composer.composed_id2column

        if 'N' not in unigram_source.available_pos:
            raise ValueError('This composer requires a noun unigram vector source')

            # vector = self._get_vector(DocumentFeature.from_string('african/J_police/N'))
            # logging.info(vector)

    def __contains__(self, feature):
        """
        Accept all adjective-noun or noun-noun phrases where we have a corpus-observed vector for the head and
        a learnt matrix (through PLSR) for the modifier
        """
        # todo expand unit tests now that we have a real composer
        if feature.type not in self.entry_types:
            # ignore non-AN features
            # print "%s is not a valid type" % feature.type
            return False

        modifier, head = feature.tokens
        assert ('J', 'N') == (modifier.pos, head.pos) or ('N', 'N') == (modifier.pos, head.pos)

        # if DocumentFeature('1-GRAM', (noun,)) not in self.unigram_source:
        if DocumentFeature.from_string(str(head)) not in self.unigram_source:
            # ignore ANs containing unknown nouns
            return False

        # ignore ANs containing unknown adjectives
        return str(modifier) in self.available_modifiers

    def __str__(self):
        return '[BaroniComposer with %d modifiers and %d heads]' % \
               (len(self.available_modifiers), len(self.unigram_source))

    def __repr__(self):
        return str(self)

    def get_vector(self, feature):
        # todo test properly
        """

        :param feature: DocumentFeature to compose, assumed to be an adjective/noun and a noun, with PoS tags
        :return:
         :rtype: 1xN scipy sparse matrix of type numpy.float64 with M stored elements in Compressed Sparse Row format,
         where N is the dimensionality of the vectors in the unigram source
        """
        modifier = str(feature.tokens[0])
        head = str(feature.tokens[1])
        phrase = '{}_{}'.format(modifier, head)
        x = self._composer.compose([(modifier, head, phrase)], self.dissect_core_space).cooccurrence_matrix.mat
        # todo could also convert to dense 1D ndarray, vector.A.ravel()
        return x


class DummyThesaurus(Thesaurus):
    """
    A thesaurus-like object which has either:
     1) a single neighbour for every possible entry, b/N
     2) a single random neighbour for every possible entry. That neighbour is chosen from the vocabulary that is
        passed in (as a dict {feature:index} )
    """
    name = 'Constant'

    def __init__(self, vocab=None, k=1, constant=True):
        self.vocab = vocab
        self.k = k
        self.constant = constant


    def __getitem__(self, feature):
        if self.constant:
            return [('b/N', 1.0)]
        else:
            if not self.vocab:
                raise ValueError('You need to provide a set of value to choose from first.')
            return [(choice(self.vocab.keys()), 1.0) for _ in range(self.k)]


    def get_vector(self):
        pass

    def to_shelf(self, *args, **kwargs):
        pass

    def __len__(self):
        return 9999999

    def __contains__(self, feature):
        return True
