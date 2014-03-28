from collections import Counter
import cPickle as pickle
import matplotlib.pyplot as plt
import pandas as pd
from thesisgenerator.plugins.bov_feature_handlers import StatsRecorder


def histogram_from_list(l, path):
    MAX_LABEL_COUNT = 20

    plt.figure()
    if type(l[0]) == str:
        # numpy's histogram doesn't like strings
        s = pd.Series(Counter(l))
        s.plot(kind='bar', rot=0)
    else:
        plt.hist(l, bins=MAX_LABEL_COUNT)

    plt.savefig(path, format='png')


def get_basic_paraphrase_statistics(stats):
    '''
    Look at the kinds of replacements that are being made at decode time
    :param stats:
    :type stats: StatsRecorder
    :return:
    :rtype:
    '''
    replacement_count = [x.available_replacements for x in stats.paraphrases]
    replacement_rank = [r for x in stats.paraphrases for r in x.ranks]
    replacement_sims = [s for x in stats.paraphrases for s in x.similarities]
    replacement_types = [feat.type for x in stats.paraphrases for feat in x.replacements]
    return replacement_count, replacement_rank, replacement_sims, replacement_types


def get_classificational_similarity_statistics(stats):
    if hasattr(stats, 'ratio'):
        replacement_ratios = {}
        for p in stats.paraphras:
            if p.original in stats.ratio:  # decode-time feature seen in training
                original_value = stats.ratio[p.original]  # log class-conditional ratio
                replacement_value = ([stats.ratio[r] for r in p.replacements])
                print p, original_value, replacement_value

    return replacement_ratios


if __name__ == '__main__':
    name = 'exp0-0'
    with open('stats%s' % name) as infile:
        stats_over_cv = pickle.load(infile)

    for i, stats in enumerate(stats_over_cv):
        data = stats.get_paraphrase_statistics()
        if not data:
            continue  # stats may be disabled for performance reasons
        for c, datatype in zip(data, ['count', 'rank', 'sim', 'feattype']):
            histogram_from_list(c,
                                'figures/%s_cv%d_%s_hist.png' % (name, i, datatype))

        try:
            # this only works for binary classifiers
            flp = stats.nb_feature_log_prob
            ratio = flp[0, :] - flp[1, :]  # P(f|c0) / P(f|c1)
            stats.ratio = {feature: ratio[index] for index, feature in stats.nb_inv_voc.items()}
        except AttributeError:
            print 'Classifier parameters unavailable'
