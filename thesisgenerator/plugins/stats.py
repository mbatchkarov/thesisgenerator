import logging
import os
from thesisgenerator.utils.misc import noop


def sum_up_token_counts(hdf_file):
    """
    Loads a pandas DataFrame from HDF storage and sums up duplicate rows.
     For example

     cat True True
     cat True True

     becomes

     cat True True 2

     The extra columns is called 'count'

    :param hdf_file: the file to load from
    :type hdf_file: str
    :param table_name: name of table (must be contained in the HDF file)
    :type table_name: str
    :return:
    :rtype: pd.DataFrame
    """
    import pandas as pd

    df = pd.read_csv(hdf_file)
    counts = df.groupby('feature').count().feature
    assert counts.sum() == df.shape[0]  # no missing rows
    df = df.drop_duplicates()
    df.set_index('feature', inplace=True)
    df['count'] = counts
    return df


class StatsRecorder(object):
    """
    Provides facilities for counting seen, unseen,
    in-thesaurus and out-of-thesaurus tokens and types
    """

    def __init__(self, prefix=None):
        self.token_counts = []
        self.paraphrases = []
        self.prefix = prefix  # store data here instead of in memory
        self.max_rows_in_memory = 1e5  # how many items to store before flushing to disk

        self.par_file = '%s.%s.csv' % (self.prefix, 'par')
        self.tc_file = '%s.%s.csv' % (self.prefix, 'tc')

        if os.path.exists(self.par_file):
            os.unlink(self.par_file)
        if os.path.exists(self.tc_file):
            os.unlink(self.tc_file)

        with open(self.tc_file, 'w') as outfile:
            outfile.write('feature,IV,IT\n')
        with open(self.par_file, 'w') as outfile:
            outfile.write('feature,available_replacements,max_replacements,'
                          'replacement1,replacement1_rank,replacement1_sim,'
                          'replacement2,replacement2_rank,replacement2_sim,replacement3,'
                          'replacement3_rank,replacement3_sim\n')

    def _flush_df_to_hdf(self, filename, data):
        if data:
            logging.info('Flushing statistics to %s', filename)

            with open(filename, 'a') as store:
                for line in data:
                    store.write(','.join(map(str, line)))
                    store.write('\n')


    def register_token(self, feature, iv, it):
        self.token_counts.append([feature.tokens_as_str(), iv, it])
        if len(self.token_counts) > self.max_rows_in_memory:
            self._flush_df_to_hdf(self.tc_file, self.token_counts)
            self.token_counts = []  # clear the chunk of data held in memory

    def consolidate_stats(self):
        self._flush_df_to_hdf(self.par_file, self.paraphrases)
        self._flush_df_to_hdf(self.tc_file, self.token_counts)
        self.token_counts = []
        self.paraphrases = []

    def register_paraphrase(self, event):
        # pad to size, making sure the right dtypes are inserted
        # introducing NaN into the table causes pandas to promote column type, which
        # results in incompatibility between the table on disk and the one in memory
        # http://pandas.pydata.org/pandas-docs/stable/gotchas.html
        while True:
            current = len(event)
            # expected = len(self.paraphrases.columns)
            expected = 12
            if current >= expected:
                break
            event.extend(['NaN', 'NaN', 'NaN'])
        self.paraphrases.append(event)
        if len(self.paraphrases) > self.max_rows_in_memory:
            self._flush_df_to_hdf(self.par_file, self.paraphrases)
            self.paraphrases = []  # clear the chunk of data held in memory


class NoopStatsRecorder(StatsRecorder):
    def __init__(self):
        pass

    register_token = consolidate_stats = register_paraphrase = get_paraphrase_statistics = noop


def get_stats_recorder(enabled, stats_file_prefix, suffix):
    if enabled and stats_file_prefix:
        return StatsRecorder(prefix='%s%s' % (stats_file_prefix, suffix))
    else:
        return NoopStatsRecorder()