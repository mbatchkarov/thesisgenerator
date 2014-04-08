import logging
import os
import pandas as pd
from thesisgenerator.utils.misc import noop


def sum_up_token_counts(hdf_file, table_name='token_counts', columns=('feature', 'IV', 'IT')):
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
    :param columns: the columns that the table is expected to have. These are set on the table
    :type columns: iterable of str
    :return:
    :rtype: pd.DataFrame
    """
    df = pd.read_hdf(hdf_file, table_name)
    df.columns = columns
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

    def __init__(self, hdf_file=None):
        self.token_counts = []
        self.paraphrases = []
        self.hdf_file = hdf_file  # store data here instead of in memory
        self.max_rows_in_memory = 1e5  # how many items to store before flushing to HDF

        if os.path.exists(self.hdf_file):
            os.unlink(self.hdf_file)

    def _flush_df_to_hdf(self, table_name, data):
        logging.info('Flushing %s statistics to %s', table_name, self.hdf_file)
        df = pd.DataFrame(data)
        with pd.get_store(self.hdf_file) as store:
            store.append(table_name, df.convert_objects())  #, min_itemsize={'values': 50, 'index': 50}
            store.flush()



    def register_token(self, feature, iv, it):
        self.token_counts.append([feature.tokens_as_str(), iv, it])
        if len(self.token_counts) > self.max_rows_in_memory:
            self._flush_df_to_hdf('token_counts', self.token_counts)
            self.token_counts = []  # clear the chunk of data held in memory

    def consolidate_stats(self):
        self._flush_df_to_hdf('paraphrases', self.paraphrases)
        self._flush_df_to_hdf('token_counts', self.token_counts)
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
            event.extend(['NONE', -1, -1.0])
        self.paraphrases.append(event)
        if len(self.paraphrases) > self.max_rows_in_memory:
            self._flush_df_to_hdf('paraphrases', self.paraphrases)
            self.paraphrases = []  # clear the chunk of data held in memory


class NoopStatsRecorder(StatsRecorder):
    def __init__(self):
        pass

    register_token = consolidate_stats = register_paraphrase = get_paraphrase_statistics = noop


def get_stats_recorder(enabled, stats_hdf_file, suffix):
    if enabled and stats_hdf_file:
        return StatsRecorder(hdf_file='%s%s' % (stats_hdf_file, suffix))
    else:
        return NoopStatsRecorder()