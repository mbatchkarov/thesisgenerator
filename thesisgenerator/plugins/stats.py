import logging
import gzip
from collections import Counter
from thesisgenerator.utils.misc import noop


class StatsRecorder(object):
    """
    Provides facilities for counting seen, unseen,
    in-thesaurus and out-of-thesaurus tokens and types
    """

    def __init__(self, prefix, stage, cv_fold, n_replacements=3):
        self.token_counts = Counter()
        self.paraphrases = Counter()
        self.prefix = prefix  # store data here instead of in memory
        self.stage = stage
        self.cv_fold = cv_fold

        self.par_file = '%s.%s.csv.gz' % (self.prefix, 'par')  # paraphrases
        self.tc_file = '%s.%s.csv.gz' % (self.prefix, 'tc')  # term counts
        self.max_paraphrases = n_replacements

        if cv_fold == 0 and stage == 'tr':
            # experiment just started, write header to output files
            with gzip.open(self.tc_file, 'wb') as outfile:
                outfile.write(bytes('# feature counts in labelled data\n', encoding='UTF8'))
                outfile.write(bytes('cv_fold,stage,feature,IV,IT,count\n', encoding='UTF8'))
            with gzip.open(self.par_file, 'wb') as outfile:
                outfile.write(bytes('# Replacements made at decode time\n', encoding='UTF8'))
                repl_header = ','.join('neigh{0},neigh{0}_sim'.format(i + 1) for i in range(n_replacements))
                header = 'cv_fold,stage,feature,available_replacements,%s,count\n' % repl_header
                outfile.write(bytes(header, encoding='UTF8'))

    def _flush_data_to_csv(self, filename, data):
        if data:
            logging.info('Flushing %s statistics for fold %d to %s', self.stage, self.cv_fold, filename)

            with gzip.open(filename, 'ab') as store:
                for value, count in data.items():
                    s = ','.join(map(str, value + (count,)))
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
        expected = 2 + 2 * self.max_paraphrases
        while len(event) < expected:
            event += ('', '')
        self.paraphrases[(self.cv_fold, self.stage) + event] += 1


class NoopStatsRecorder(StatsRecorder):
    def __init__(self):
        pass

    register_token = consolidate_stats = register_paraphrase = get_paraphrase_statistics = noop


def get_stats_recorder(enabled, stats_file_prefix, stage, cv_fold, n_replacements):
    if enabled and stats_file_prefix:
        return StatsRecorder(stats_file_prefix, stage, cv_fold, n_replacements=n_replacements)
    else:
        return NoopStatsRecorder()
