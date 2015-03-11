import logging
import gzip
from collections import Counter
from thesisgenerator.utils.misc import noop
import pandas as pd


def sum_up_token_counts(filename):
    """
    Loads a pandas DataFrame from HDF storage and sums up duplicate rows.
     For example

     cat True True
     cat True True

     becomes

     cat True True 2

     The extra columns is called 'count'

    :param filename: the file to load from
    :type filename: str
    :rtype: pd.DataFrame
    """
    df = pd.read_csv(filename, sep=', ')
    counts = df.groupby('feature').count()
    assert counts.IV.sum() == counts.IT.sum() == df.shape[0]  # no missing rows
    df = df.drop_duplicates()
    df.set_index('feature', inplace=True)
    df['count'] = counts.IV
    return df


class StatsRecorder(object):
    """
    Provides facilities for counting seen, unseen,
    in-thesaurus and out-of-thesaurus tokens and types
    """

    def __init__(self, prefix, stage, cv_fold):
        self.token_counts = Counter()
        self.paraphrases = Counter()
        self.prefix = prefix  # store data here instead of in memory
        self.stage = stage
        self.cv_fold = cv_fold

        self.par_file = '%s.%s.csv.gz' % (self.prefix, 'par')
        self.tc_file = '%s.%s.csv.gz' % (self.prefix, 'tc')

        if cv_fold == 0 and stage == 0:
            # experiment just started, write header to output files
            with gzip.open(self.tc_file, 'wb') as outfile:
                outfile.write(bytes('cv_fold, stage, feature, IV, IT, count\n', encoding='UTF8'))
            with gzip.open(self.par_file, 'wb') as outfile:
                header = 'cv_fold, stage, feature, available_replacements, ' \
                         'replacement1, replacement1_sim, ' \
                         'replacement2, replacement2_sim, ' \
                         'replacement3, replacement3_sim, count\n'
                outfile.write(bytes(header, encoding='UTF8'))

    def _flush_data_to_csv(self, filename, data):
        if data:
            logging.info('Flushing %s statistics for fold %d to %s', self.stage, self.cv_fold, filename)

            with gzip.open(filename, 'ab') as store:
                for value, count in data.items():
                    s = ', '.join(map(str, value + (count,)))
                    store.write(bytes(s, encoding='UTF8'))
                    store.write(bytes('\n', encoding='UTF8'))


    def register_token(self, feature, iv, it):
        self.token_counts[(self.cv_fold, self.stage, feature.tokens_as_str(), int(iv), int(it))] += 1

    def consolidate_stats(self):
        self._flush_data_to_csv(self.par_file, self.paraphrases)
        self._flush_data_to_csv(self.tc_file, self.token_counts)
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
            expected = 8
            if current >= expected:
                break
            event += ('NaN', 'NaN')
        self.paraphrases[(self.cv_fold, self.stage) + event] += 1
        # if len(self.paraphrases) > self.max_rows_in_memory:
        # self._flush_data_to_csv(self.par_file, self.paraphrases)
        #     self.paraphrases = []  # clear the chunk of data held in memory


class NoopStatsRecorder(StatsRecorder):
    def __init__(self):
        pass

    register_token = consolidate_stats = register_paraphrase = get_paraphrase_statistics = noop


def get_stats_recorder(enabled, stats_file_prefix, stage, cv_fold):
    if enabled and stats_file_prefix:
        return StatsRecorder(stats_file_prefix, stage, cv_fold)
    else:
        return NoopStatsRecorder()