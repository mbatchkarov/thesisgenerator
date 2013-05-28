from collections import deque
import logging
from thesisgenerator.plugins.thesaurus_loader import get_all_thesauri


def get_stats_recorder(enabled=False):
    return StatsRecorder() if enabled else NoopStatsRecorder()


def get_token_handler(replace_all, use_signifier_only):
    if replace_all:
        return ReplaceAllFeatureHandler()
    else:
        if use_signifier_only:
            return BaseFeatureHandler()
        else:
            return SignifierSignifiedFeatureHandler()


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
            logging.getLogger().debug('IV IT token %s' % token)
        elif iv and not it:
            self.iv_oot.append(token)
            logging.getLogger().debug('IV OOT token %s' % token)
        elif not iv and it:
            self.oov_it.append(token)
            logging.getLogger().debug('OOV IT token %s' % token)
        else:
            self.oov_oot.append(token)
            logging.getLogger().debug('OOV OOT token %s' % token)
            # self.all_types.add(token)
            # self.num_tokens += 1

    def print_coverage_stats(self):
        logging.getLogger().info('Vectorizer: '
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
        # logging.getLogger().debug('IV IT %s'% self.iv_it)
        # logging.getLogger().debug('IV 00T %s' % self.iv_oot)
        # logging.getLogger().debug('OOV IT %s' % self.oov_it)
        # logging.getLogger().debug('OOV OOT %s' % self.oov_oot)


class NoopStatsRecorder(StatsRecorder):
    def register_token(self, token, iv, it):
        pass

    def print_coverage_stats(self):
        pass


class BaseFeatureHandler():
    """
    Handles features the way standard Naive Bayes does:
        - in vocabulary, in thesaurus: only insert feature itself
        - IV,OOT: feature itself
        - OOV, IT: ignore feature
        - OOV, OOT: ignore feature
    """

    def _insert_feature_only(self, doc_id, doc_id_indices, document_term,
                             term_indices, term_index_in_vocab, values, count):
        logging.getLogger().debug(
            'Inserting feature in doc %d: %s' % (doc_id, document_term))
        doc_id_indices.append(doc_id)
        term_indices.append(term_index_in_vocab)
        values.append(count)

    def _ignore_feature(self, doc_id, document_term):
        logging.getLogger().debug(
            'Ignoring feature in doc %d: %s' % (doc_id, document_term))

    def _insert_thesaurus_neighbours(self, doc_id, doc_id_indices,
                                     document_term, term_indices,
                                     values, vocabulary):
        """
        Replace term with its k nearest neighbours from the thesaurus
        """

        # logger.info below demonstrates that unseen words exist,
        # i.e. vectorizer is not reducing the test set to the
        # training vocabulary
        neighbours = get_all_thesauri().get(document_term)

        # if there are any neighbours filter the list of
        # neighbours so that it contains only pairs where
        # the neighbour has been seen
        neighbours = [(neighbour, sim) for neighbour, sim in
                      neighbours if
                      neighbour in vocabulary] if neighbours \
            else []
        for neighbour, sim in neighbours:
            logging.getLogger().debug(
                'Replacement. Doc %d: %s --> %s, '
                'sim = %f' % (
                    doc_id, document_term, neighbour, sim))
            # todo the document may already contain the feature we
            # are about to insert into it,
            # a mergin strategy is required,
            # e.g. what do we do if the document has the word X
            # in it and we encounter X again. By default,
            # scipy uses addition
            doc_id_indices.append(doc_id)
            term_indices.append(vocabulary.get(neighbour))
            values.append(sim)

    def handle_IV_IT_feature(self, doc_id, doc_id_indices, document_term,
                             term_indices, term_index_in_vocab, values,
                             count, vocabulary):
        self._insert_feature_only(doc_id, doc_id_indices, document_term,
                                  term_indices, term_index_in_vocab, values,
                                  count)

    def handle_IV_OOT_feature(self, doc_id, doc_id_indices, document_term,
                              term_indices, term_index_in_vocab, values, count,
                              vocabulary):
        self._insert_feature_only(doc_id, doc_id_indices, document_term,
                                  term_indices, term_index_in_vocab, values,
                                  count)

    def handle_OOV_IT_feature(self, doc_id, doc_id_indices, document_term,
                              term_indices, term_index_in_vocab, values, count,
                              vocabulary):
        self._ignore_feature(doc_id, document_term)

    def handle_OOV_OOT_feature(self, doc_id, doc_id_indices, document_term,
                               term_indices, term_index_in_vocab, values,
                               count, vocabulary):
        self._ignore_feature(doc_id, document_term)


class SignifierSignifiedFeatureHandler(BaseFeatureHandler):
    """
    Handles features the way standard Naive Bayes does, except
        - OOV, IT: insert K neighbours from thesaurus instead of ignoring the
        feature
    """

    def handle_OOV_IT_feature(self, doc_id, doc_id_indices, document_term,
                              term_indices, term_index_in_vocab, values, count,
                              vocabulary):
        self._insert_thesaurus_neighbours(doc_id, doc_id_indices,
                                          document_term, term_indices,
                                          values, vocabulary)


class ReplaceAllFeatureHandler(BaseFeatureHandler):
    """
    Handles features the way standard Naive Bayes does, except
        - OOV, IT: insert K neighbours from thesaurus
        - IV, IT: insert K neighbours from thesaurus

        Note: no token can ever be IV and OOT in this setting
    """

    def handle_IV_IT_feature(self, doc_id, doc_id_indices, document_term,
                             term_indices, term_index_in_vocab, values, count,
                             vocabulary):
        self._insert_thesaurus_neighbours(doc_id, doc_id_indices,
                                          document_term, term_indices,
                                          values, vocabulary)

    handle_OOV_IT_feature = handle_IV_IT_feature

    def handle_IV_OOT_feature(self, doc_id, doc_id_indices, document_term,
                              term_indices, term_index_in_vocab, values, count,
                              vocabulary):
        raise Exception('This must never be reached because set(vocabulary)=='
                        ' set(thesaurus.keys()). It is not possible for a '
                        'feature to be IV and OOT')

