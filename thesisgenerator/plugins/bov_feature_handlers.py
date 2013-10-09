from collections import deque
import logging
from thesisgenerator.utils.reflection_utils import get_named_object


def get_stats_recorder(enabled=False):
    return StatsRecorder() if enabled else NoopStatsRecorder()


def get_token_handler(handler_name, k, transformer_name, vector_source):
    # k- parameter for _paraphrase
    # sim_transformer- callable that transforms the raw sim scores in
    # _paraphrase
    # todo replace k with a named object
    handler = get_named_object(handler_name)
    transformer = get_named_object(transformer_name)
    logging.info('Returning token handler %s (k=%s, sim transformer=%s)' % (
        handler,
        k,
        transformer))
    return handler(k, transformer, vector_source)


class StatsRecorder(object):
    """
    Provides facilities for counting seen, unseen,
    in-thesaurus and out-of-thesaurus tokens and types
    """

    def __init__(self):
        self.iv_it = deque()
        self.iv_oot = deque()
        self.oov_it = deque()
        self.oov_oot = deque()

    def register_token(self, token, iv, it):
        if iv and it:
            self.iv_it.append(token)
            logging.debug('IV IT token {}'.format(token))
        elif iv and not it:
            self.iv_oot.append(token)
            logging.debug('IV OOT token {}'.format(token))
        elif not iv and it:
            self.oov_it.append(token)
            logging.debug('OOV IT token {}'.format(token))
        else:
            self.oov_oot.append(token)
            logging.debug('OOV OOT token {}'.format(token))

    def print_coverage_stats(self):
        logging.info('Vectorizer: '
                     'IV IT tokens: %d, '
                     'IV OOT tokens: %d, '
                     'OOV IT tokens: %d, '
                     'OOV OOT tokens: %d, '
                     'IV IT types: %d, '
                     'IV OOT types: %d, '
                     'OOV IT types: %d, '
                     'OOV OOT types: %d ' % (
                         len(self.iv_it),
                         len(self.iv_oot),
                         len(self.oov_it),
                         len(self.oov_oot),
                         len(set(self.iv_it)),
                         len(set(self.iv_oot)),
                         len(set(self.oov_it)),
                         len(set(self.oov_oot))))
        # logging.debug('IV IT %s'% self.iv_it)
        # logging.debug('IV 00T %s' % self.iv_oot)
        # logging.debug('OOV IT %s' % self.oov_it)
        # logging.debug('OOV OOT %s' % self.oov_oot)


class NoopStatsRecorder(StatsRecorder):
    def register_token(self, token, iv, it):
        pass

    def print_coverage_stats(self):
        pass


def _insert_feature_only(doc_id, feature, feature_index_in_vocab, j_indices, values):
    logging.debug('Inserting feature in doc %d: %s' % (doc_id, feature))
    j_indices.append(feature_index_in_vocab)
    values.append(1)


def _ignore_feature(doc_id, document_term):
    logging.debug('Ignoring feature in doc %d: %s' % (
        doc_id, document_term))
    pass


def _paraphrase(doc_id, feature, feature_index_in_vocab, vocabulary, j_indices,
                vector_source, k, sim_transformer, values):
    """
    Replaces term with its k nearest neighbours from the thesaurus

    Parameters
    ----------
    neighbour_source : callable, returns a thesaurus-like object (a list of
      (neighbour, sim) tuples, sorted by highest sim first,
      acts as a defaultdict(list) ). The callable takes one parameter for
      compatibility purposes- one of the possible callables I want to
      use here requires access to the vocabulary.
       The default behaviour is to return a callable pointing to the
       currently loaded thesaurus.
    """

    #neighbours = thesaurus(vocabulary)[document_term]
    neighbours = vector_source.get_nearest_neighbours(feature)

    # if there are any neighbours filter the list of
    # neighbours so that it contains only pairs where
    # the neighbour has been seen
    neighbours = [(neighbour, sim) for neighbour, sim in neighbours
                  if neighbour in vocabulary]

    logging.debug('Using %d/%d IV neighbours' % (k, len(neighbours)))
    for neighbour, sim in neighbours[:k]:
        logging.debug('Replacement. Doc %d: %s --> %s, sim = %f' % (
            doc_id, feature, neighbour, sim))

        # todo the document may already contain the feature we
        # are about to insert into it,
        # a mergin strategy is required,
        # e.g. what do we do if the document has the word X
        # in it and we encounter X again. By default,
        # scipy uses addition
        #doc_id_indices.append(doc_id)
        j_indices.append(vocabulary.get(neighbour))
        values.append(sim_transformer(sim))


class BaseFeatureHandler():
    """
    Handles features the way standard Naive Bayes does:
        - in vocabulary, in thesaurus: only insert feature itself
        - IV,OOT: feature itself
        - OOV, IT: ignore feature
        - OOV, OOT: ignore feature
    """

    def __init__(self, k, sim_transformer, vector_source):
        # contructor takes parameters for compatibility with others
        pass

    def handle_IV_IT_feature(self, doc_id, feature, feature_index_in_vocab, vocabulary, j_indices, values):
        _insert_feature_only(doc_id, feature, feature_index_in_vocab, j_indices, values)

    def handle_IV_OOT_feature(self, doc_id, feature, feature_index_in_vocab, vocabulary, j_indices, values):
        _insert_feature_only(doc_id, feature, feature_index_in_vocab, j_indices, values)

    def handle_OOV_IT_feature(self, doc_id, feature, feature_index_in_vocab, vocabulary, j_indices, values):
        _ignore_feature(doc_id, feature)

    def handle_OOV_OOT_feature(self, doc_id, feature, feature_index_in_vocab, vocabulary, j_indices, values):
        _ignore_feature(doc_id, feature)


class SignifierSignifiedFeatureHandler(BaseFeatureHandler):
    """
    Handles features the way standard Naive Bayes does, except
        - OOV, IT: insert the first K IV neighbours from thesaurus instead of
        ignoring the feature
    """

    def __init__(self, k, sim_transformer, vector_source):
        self.k = k
        self.sim_transformer = sim_transformer
        self.vector_source = vector_source

    def handle_OOV_IT_feature(self, doc_id, feature, feature_index_in_vocab, vocabulary, j_indices, values):
        _paraphrase(doc_id, feature, feature_index_in_vocab,
                    vocabulary, j_indices, self.vector_source,
                    self.k, self.sim_transformer, values)


class SignifiedOnlyFeatureHandler(BaseFeatureHandler):
    """
    Ignores all OOT features and inserts the first K IV neighbours from
    thesaurus for all IT features
    """

    def __init__(self, k, sim_transformer, vector_source):
        self.k = k
        self.sim_transformer = sim_transformer
        self.vector_source = vector_source

    def handle_OOV_IT_feature(self, doc_id, feature, feature_index_in_vocab, vocabulary, j_indices, values):
        _paraphrase(doc_id, feature, feature_index_in_vocab,
                    vocabulary, j_indices, self.vector_source,
                    self.k, self.sim_transformer, values)

    handle_IV_IT_feature = handle_OOV_IT_feature

    def handle_IV_OOT_feature(self, doc_id, feature, feature_index_in_vocab, vocabulary, j_indices, values):
        _ignore_feature(doc_id, feature)


class SignifierRandomBaselineFeatureHandler(SignifiedOnlyFeatureHandler):
    """
    Ignores all OOT features and inserts K random IV tokens for all IT features
    """

    def __init__(self, k, sim_transformer, vector_source):
        self.k = k
        self.sim_transformer = sim_transformer
        self.vector_source = vector_source

    def handle_OOV_IT_feature(self, doc_id, feature, feature_index_in_vocab, vocabulary, j_indices, values):
        _paraphrase(doc_id, feature, feature_index_in_vocab,
                    vocabulary, j_indices, self.vector_source,
                    self.k, self.sim_transformer, values)

    handle_IV_IT_feature = handle_OOV_IT_feature