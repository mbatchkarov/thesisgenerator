import logging
from thesisgenerator.plugins.thesaurus_loader import get_all_thesauri


def get_handler(replace_all, vocab_from_thes):
    if replace_all:
        return ReplaceAllFeatureHandler()
    else:
        if vocab_from_thes:
            return VocabFromThesaurusBaselineFeatureHandler()
        else:
            return BaseFeatureHandler()


class StatsRecordingFeatureHandlerMixin(object):
    def __init__(self):
        self.recording = False

    def begin_stats_recording(self):
        # how many tokens are there/ are unknown/ have been replaced
        self.num_tokens, self.unknown_tokens = 0, 0
        self.found_tokens, self.replaced_tokens = 0, 0
        self.all_types = set()
        self.unknown_types = set()
        self.found_types = set()
        self.replaced_types = set()

    def register_token(self, document_term):
        # todo this needs to be conditional on some global switch
        if not self.recording:
            self.begin_stats_recording()
            self.recording = True

        self.all_types.add(document_term)
        self.num_tokens += 1

    def print_coverage_stats(self):
        logging.getLogger('root').info('Vectorizer: '
                                       'Total tokens: %d, '
                                       'Unknown tokens: %d, '
                                       'Found tokens: %d, '
                                       'Replaced tokens: %d, '
                                       'Total types: %d, '
                                       'Unknown types: %d,  '
                                       'Found types: %d, '
                                       'Replaced types: %d' % (
                                           self.num_tokens,
                                           self.unknown_tokens,
                                           self.found_tokens,
                                           self.replaced_tokens,
                                           len(self.all_types),
                                           len(self.unknown_types),
                                           len(self.found_types),
                                           len(self.replaced_types)))


class BaseFeatureHandler(StatsRecordingFeatureHandlerMixin):
    def _insert_feature_only(self, doc_id, doc_id_indices, document_term,
                             term_indices, term_index_in_vocab, values, count):
        logging.getLogger('root').debug(
            'Known token in doc %d: %s' % (doc_id, document_term))
        doc_id_indices.append(doc_id)
        term_indices.append(term_index_in_vocab)
        values.append(count)

    def _ignore_feature(self, doc_id, document_term):
        logging.getLogger('root').debug(
            'Non-thesaurus token in doc %d: %s' % (doc_id, document_term))

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
        if neighbours:
            self.found_tokens += 1
            self.found_types.add(document_term)
            logging.getLogger('root').debug('Found thesaurus entry '
                                            'for %s' % document_term)

        neighbours = [(neighbour, sim) for neighbour, sim in
                      neighbours if
                      neighbour in vocabulary] if neighbours \
            else []
        if len(neighbours) > 0:
            self.replaced_tokens += 1
            self.replaced_types.add(document_term)
        for neighbour, sim in neighbours:
            logging.getLogger('root').debug(
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


class VocabFromThesaurusBaselineFeatureHandler(BaseFeatureHandler):
    pass


class ReplaceAllFeatureHandler(BaseFeatureHandler):
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


