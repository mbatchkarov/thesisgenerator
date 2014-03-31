from collections import deque
import logging
import os
from thesisgenerator.utils.misc import noop
from thesisgenerator.utils.reflection_utils import get_named_object
import pandas as pd


def get_stats_recorder(enabled, stats_hdf_file, suffix):
    f = '%s%s' % (stats_hdf_file, suffix)
    return StatsRecorder(hdf_file=f) if enabled and stats_hdf_file else NoopStatsRecorder()


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


class LexicalReplacementEvent(object):
    def __init__(self, original, max_replacements, available_replacements, replacements, ranks, similarities):
        self.original = original
        self.max_replacements = max_replacements
        self.available_replacements = available_replacements
        self.replacements = replacements
        self.ranks = ranks
        self.similarities = similarities

    def __str__(self):
        return '%s (%d/%d available) --> %s, %s, %s' % (self.original, self.available_replacements,
                                                        self.max_replacements, self.replacements,
                                                        self.ranks, self.similarities)


class StatsRecorder(object):
    """
    Provides facilities for counting seen, unseen,
    in-thesaurus and out-of-thesaurus tokens and types
    """

    def __init__(self, hdf_file=None):
        self.token_counts = pd.DataFrame(columns=('feature', 'count', 'IV', 'IT'))
        self.token_counts.set_index('feature', inplace=True)
        self.paraphrases = pd.DataFrame(columns=('feature', 'available_replacements', 'max_replacements',
                                                 'replacement1', 'replacement1_rank', 'replacement1_sim',
                                                 'replacement2', 'replacement2_rank', 'replacement2_sim',
                                                 'replacement3', 'replacement3_rank', 'replacement3_sim'))
        self.max_rows_in_memory = 1e20  # hold all data in memory

        if hdf_file:
            self.hdf_file = hdf_file  # store data here instead of in memory
            self.max_rows_in_memory = 2  # how many items to store before flushing to HDF

            if os.path.exists(self.hdf_file):
                os.unlink(self.hdf_file)

    def _flush_df_to_hdf(self, table_name, table):
        with pd.get_store(self.hdf_file) as store:
            table.fillna(-1)
            store.append(table_name, table.convert_objects(),
                         min_itemsize={'values': 50, 'index': 50})

    def register_token(self, feature, iv, it):
        s = feature.tokens_as_str()
        try:
            # if feature has been seen before increment count
            row = self.token_counts.loc[s]
            if map(bool, row.tolist()[1:]) == [iv, it]:
                # this increment affects all columns in the given row, but that's OK
                self.token_counts.loc[s, 'count'] += 1
            else:
                raise ValueError('The same feature seen with different IV/IT values, this is odd.')
        except KeyError:
            # token not known yet
            new_df = pd.DataFrame([[1, iv, it]], columns=self.token_counts.columns, index=[s])
            self.token_counts = pd.concat([self.token_counts, new_df])

        if self.token_counts.shape[0] > self.max_rows_in_memory and self.hdf_file:
            self._flush_df_to_hdf('token_counts', self.token_counts)
            self.token_counts = self.token_counts[0:0]  # clear the chunk of data held in memory

    def consolidate_stats(self):
        self._flush_df_to_hdf('token_counts', self.token_counts)
        reader = pd.read_hdf(self.hdf_file, 'token_counts', chunksize=1e5)
        for chunk in reader:  # read the table bit by bit to save memory
            tmp = pd.concat([self.token_counts, chunk])
            self.token_counts = tmp.groupby(tmp.index).sum()  # add up the occurrences of each feature
        with pd.get_store(self.hdf_file) as store:
            store['token_counts'] = self.token_counts

        self._flush_df_to_hdf('paraphrases', self.paraphrases)

    def register_paraphrase(self, event):
        # pad to size, making sure the right dtypes are inserted
        # introducing NaN into the table causes pandas to promote column type, which
        # results in incompatibility between the table on disk and the one in memory
        # http://pandas.pydata.org/pandas-docs/stable/gotchas.html
        while True:
            current = len(event)
            expected = len(self.paraphrases.columns)
            if current >= expected:
                break
            event.extend(['NONE', -1, -1.0])
        new_df = pd.DataFrame([event],
                              columns=self.paraphrases.columns)
        self.paraphrases = pd.concat([self.paraphrases, new_df])
        if self.paraphrases.shape[0] > self.max_rows_in_memory and self.hdf_file:
            self._flush_df_to_hdf('paraphrases', self.paraphrases)
            self.paraphrases = self.paraphrases[0:0]


class NoopStatsRecorder(StatsRecorder):
    register_token = consolidate_stats = register_paraphrase = get_paraphrase_statistics = noop


class BaseFeatureHandler():
    """
    Handles features the way standard Naive Bayes does:
        - in vocabulary, in thesaurus: only insert feature itself
        - IV,OOT: feature itself
        - OOV, IT: ignore feature
        - OOV, OOT: ignore feature
    """

    def __init__(self, *args):
        pass

    def handle_IV_IT_feature(self, **kwargs):
        self._insert_feature_only(**kwargs)

    def handle_IV_OOT_feature(self, **kwargs):
        self._insert_feature_only(**kwargs)

    def handle_OOV_IT_feature(self, **kwargs):
        self._ignore_feature(**kwargs)

    def handle_OOV_OOT_feature(self, **kwargs):
        self._ignore_feature(**kwargs)


    def _insert_feature_only(self, feature_index_in_vocab, j_indices, values, **kwargs):
        #logging.debug('Inserting feature in doc %d: %s', doc_id, feature)
        j_indices.append(feature_index_in_vocab)
        values.append(1)

    def _ignore_feature(self, doc_id, feature, **kwargs):
        #logging.debug('Ignoring feature in doc %d: %s', doc_id, feature)
        pass

    def _paraphrase(self, feature, vocabulary, j_indices, values, stats, **kwargs):
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

        #logging.debug('Paraphrasing %r in doc %d', feature, doc_id)
        neighbours = self.vector_source.get_nearest_neighbours(feature)

        # if there are any neighbours filter the list of
        # neighbours so that it contains only pairs where
        # the neighbour has been seen
        neighbours = [(neighbour, rank, sim) for rank, (neighbour, sim) in enumerate(neighbours)
                      if neighbour in vocabulary]
        k, available_neighbours = self.k, len(neighbours)
        event = [feature.tokens_as_str(), available_neighbours, self.k]

        for neighbour, rank, sim in neighbours[:self.k]:
            # todo the document may already contain the feature we
            # are about to insert into it,
            # a merging strategy is required,
            # e.g. what do we do if the document has the word X
            # in it and we encounter X again. By default,
            # scipy uses addition
            #doc_id_indices.append(doc_id)
            j_indices.append(vocabulary.get(neighbour))
            values.append(self.sim_transformer(sim))

            # track the event
            event.extend([neighbour.tokens_as_str(), rank, sim])
        stats.register_paraphrase(event)


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

    def handle_OOV_IT_feature(self, **kwargs):
        self._paraphrase(**kwargs)


class SignifiedOnlyFeatureHandler(BaseFeatureHandler):
    """
    Ignores all OOT features and inserts the first K IV neighbours from
    thesaurus for all IT features
    """

    def __init__(self, k, sim_transformer, vector_source):
        self.k = k
        self.sim_transformer = sim_transformer
        self.vector_source = vector_source

    def handle_OOV_IT_feature(self, **kwargs):
        self._paraphrase(**kwargs)

    handle_IV_IT_feature = handle_OOV_IT_feature

    def handle_IV_OOT_feature(self, **kwargs):
        self._ignore_feature(**kwargs)


class SignifierRandomBaselineFeatureHandler(SignifiedOnlyFeatureHandler):
    """
    Ignores all OOT features and inserts K random IV tokens for all IT features
    """

    def __init__(self, k, sim_transformer, vector_source):
        self.k = k
        self.sim_transformer = sim_transformer
        self.vector_source = vector_source

    def handle_OOV_IT_feature(self, **kwargs):
        self._paraphrase(**kwargs)

    handle_IV_IT_feature = handle_OOV_IT_feature