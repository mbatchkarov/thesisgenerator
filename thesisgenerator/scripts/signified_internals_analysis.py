from collections import Counter, namedtuple
import cPickle as pickle
from itertools import chain
from operator import add
from discoutils.tokens import DocumentFeature
import matplotlib.pyplot as plt
import pandas as pd
from thesisgenerator.plugins.stats import sum_up_token_counts
from collections import defaultdict
import numpy as np


def histogram_from_list(l, path):
    MAX_LABEL_COUNT = 40
    print 'Running histogram'
    plt.figure()
    if type(l[0]) == str:
        # numpy's histogram doesn't like strings
        s = pd.Series(Counter(l))
        s.plot(kind='bar', rot=0)
    else:
        plt.hist(l, bins=MAX_LABEL_COUNT)

    plt.savefig(path, format='png')


def _train_time_counts(fname):
    # BASIC STATISTICS AT TRAINING TIME
    df = sum_up_token_counts(fname)
    vocab = set(df.index.tolist())
    df_it = df[df['IT'] > 0]
    return namedtuple('TrainCount', 'tokens, types, it_tokens, it_types')(tokens=df['count'].sum(),
                                                                          types=df.shape[0],
                                                                          it_tokens=df_it['count'].sum(),
                                                                          it_types=df_it.shape[0])


def _decode_time_counts(fname):
    # BASIC STATISTICS AT DECODE TIME
    df = sum_up_token_counts(fname)
    df_it = df[df['IT'] > 0]
    df_iv = df[df['IV'] > 0]

    CountsResult = namedtuple('CountsResult', 'tokens types iv_types it_types iv it iv_it iv_oot oov_it oov_oot')
    return CountsResult(
        tokens=df['count'].sum(),
        types=df.shape[0],
        it=df_it['count'].sum(),
        it_types=df_it.shape[0],
        iv=df_iv['count'].sum(),
        iv_types=df_iv.shape[0],
        iv_it=df[(df['IT'] > 0) & (df['IV'] > 0)]['count'].sum(),
        oov_it=df[(df['IT'] == 0) & (df['IV'] > 0)]['count'].sum(),
        iv_oot=df[(df['IT'] > 0) & (df['IV'] == 0)]['count'].sum(),
        oov_oot=df[(df['IT'] == 0) & (df['IV'] == 0)]['count'].sum()
    )


def _analyse_replacement_ranks_and_sims(df, fname):
    # BASIC STATISTICS ABOUT REPLACEMENTS (RANK IN NEIGHBOURS LIST AND SIM OT ORIGINAL)
    res = {}
    for statistic in ['rank', 'sim']:
        data = []
        for i in range(1, 4):
            values = df['replacement%d_%s' % (i, statistic)]
            counts = df['count']
            for value, count in zip(values, counts):
                if value != 'NONE' and value != -1:  #
                    data.extend([value] * count)
        # histogram_from_list(data, 'figures/%s_%s_hist.png' % (fname, statistic))
        res[statistic] = data
    ReplacementsResult = namedtuple('ReplacementsResult', 'rank sim')
    return ReplacementsResult(**res)


def _analyse_replacements(fname):
    df = pd.read_hdf(fname, 'paraphrases')
    df.columns = ('feature', 'available_replacements', 'max_replacements',
                  'replacement1', 'replacement1_rank', 'replacement1_sim',
                  'replacement2', 'replacement2_rank', 'replacement2_sim',
                  'replacement3', 'replacement3_rank', 'replacement3_sim')
    counts = df.groupby('feature').count().feature
    assert counts.sum() == df.shape[0]  # no missing rows
    df = df.drop_duplicates()
    df.set_index('feature', inplace=True)
    df['count'] = counts

    #####################################################################
    # ANALYSE CLASS-CONDITIONAL PROBABILITY OF REPLACEMENTS
    #####################################################################

    with open('%s.pickle' % fname) as infile:
        stats = pickle.load(infile)
    try:
        flp = stats.nb_feature_log_prob
        # this only works for binary classifiers and IV-IT features
        ratio = flp[0, :] - flp[1, :]  # P(f|c0) / P(f|c1) in log space
        # high positive value mean strong association with class 0, very negative means the opposite
        scores = {feature: ratio[index] for index, feature in stats.nb_inv_voc.items()}
    except AttributeError:
        print 'Classifier parameters unavailable'
        return None

    replacement_scores = defaultdict(int)
    for f in df.index:  # this contains all IT features, regardless of whether they were IV or IT
        try:
            orig_score = scores[DocumentFeature.from_string(f)]
        except KeyError:
            # this feature was not in the original vocabulary, we don't know it class conditional probs
            continue

        repl_count = df.ix[f]['count']
        if repl_count > 0:
            for i in range(1, 4):
                replacement = df.ix[f]['replacement%d' % i]
                repl_sim = df.ix[f]['replacement%d_sim' % i]  # they should all be the same
                if repl_sim > 0:
                    # -1 signifies no replacement has been found
                    repl_score = repl_sim * scores[DocumentFeature.from_string(replacement)]
                    replacement_scores[(orig_score, repl_score)] += repl_count

    return _analyse_replacement_ranks_and_sims(df, fname), replacement_scores


def _print_counts_data(train_counts, title):
    print '----------------------'
    print '| %s time statistics:' % title
    for field in train_counts[0]._fields:
        print '| %s: mean %f, std %f' % ( field,
                                        np.mean([x._asdict()[field] for x in train_counts]),
                                        np.std([x._asdict()[field] for x in train_counts]))
    print '----------------------'


def do_work():
    train_counts, decode_counts = [], []
    basic_stats, replacement_scores = [], []
    for cv_fold in [0, 1]:
        fname = 'stats-exp0-0-cv%d-tr' % cv_fold
        train_counts.append(_train_time_counts(fname))

        fname = 'stats-exp0-0-cv%d-ev' % cv_fold
        decode_counts.append(_decode_time_counts(fname))
        a, b = _analyse_replacements(fname)
        basic_stats.append(a)
        replacement_scores.append(b)

    # COLLATE AND AVERAGE STATS OVER CROSSVALIDATION
    histogram_from_list(list(chain.from_iterable(x.rank for x in basic_stats)), 'figures/stats-exp0-0-repl-ranks.png')
    histogram_from_list(list(chain.from_iterable(x.sim for x in basic_stats)), 'figures/stats-exp0-0-repl-sims.png')

    _print_counts_data(train_counts, 'Train')
    _print_counts_data(decode_counts, 'Decode')


    replacement_scores = reduce(add, (Counter(x) for x in replacement_scores))
    if replacement_scores:
        # sometimes there may not be any IV-IT features at decode time
        x = []
        y = []
        thickness = []
        for (orig_value, repl_value), repl_count in replacement_scores.iteritems():
            y.append(repl_value)
            x.append(orig_value)
            thickness.append(repl_count)
        plt.figure()
        plt.scatter(x, y, thickness)
        plt.hlines(0, min(x), max(x))
        plt.vlines(0, min(y), max(y))
        plt.xlabel('Class association of decode-time feature')
        plt.ylabel('Class association of replacements')
        plt.savefig('figures/stats-exp0-0-NB-scores.png', format='png')


if __name__ == '__main__':
    do_work()

