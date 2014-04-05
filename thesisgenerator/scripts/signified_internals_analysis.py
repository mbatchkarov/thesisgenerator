from collections import Counter, namedtuple
import cPickle as pickle
from itertools import chain, groupby
import logging
from operator import add, itemgetter
from discoutils.tokens import DocumentFeature
import matplotlib.pyplot as plt
import pandas as pd
from thesisgenerator.plugins.stats import sum_up_token_counts
from collections import defaultdict
import numpy as np


def histogram_from_list(l, path):
    MAX_LABEL_COUNT = 40
    logging.info('Running histogram')
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
    logging.info('Traing time token counts')
    df = sum_up_token_counts(fname)
    vocab = set(df.index.tolist())
    df_it = df[df['IT'] > 0]
    return namedtuple('TrainCount', 'tokens, types, it_tokens, it_types')(tokens=df['count'].sum(),
                                                                          types=df.shape[0],
                                                                          it_tokens=df_it['count'].sum(),
                                                                          it_types=df_it.shape[0])


def _decode_time_counts(fname):
    # BASIC STATISTICS AT DECODE TIME
    logging.info('Decode time token counts')
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


def _analyse_replacement_ranks_and_sims(df):
    # BASIC STATISTICS ABOUT REPLACEMENTS (RANK IN NEIGHBOURS LIST AND SIM OT ORIGINAL)
    logging.info('Basic replacements stats')
    res = {}
    for statistic in ['rank', 'sim']:
        data = []
        for i in range(1, 4):
            values = df['replacement%d_%s' % (i, statistic)]
            counts = df['count']
            for value, count in zip(values, counts):
                if value > 0:  # filter out NaN-s
                    data.extend([value] * count)
        # histogram_from_list(data, 'figures/%s_%s_hist.png' % (fname, statistic))
        res[statistic] = data
    ReplacementsResult = namedtuple('ReplacementsResult', 'rank sim')
    return ReplacementsResult(**res)


def _analyse_replacements(paraphrases_file, pickle_file):
    logging.info('Advanced replacements stats')
    df = pd.read_csv(paraphrases_file, sep=', ')
    counts = df.groupby('feature').count().feature
    assert counts.sum() == df.shape[0]  # no missing rows
    df = df.drop_duplicates()
    df.set_index('feature', inplace=True)
    df['count'] = counts

    #####################################################################
    # ANALYSE CLASS-CONDITIONAL PROBABILITY OF REPLACEMENTS
    #####################################################################

    with open(pickle_file) as infile:
        stats = pickle.load(infile)
    try:
        flp = stats.nb_feature_log_prob
        # this only works for binary classifiers and IV-IT features
        ratio = flp[0, :] - flp[1, :]  # P(f|c0) / P(f|c1) in log space
        # high positive value mean strong association with class 0, very negative means the opposite
        scores = {feature: ratio[index] for index, feature in stats.nb_inv_voc.items()}
    except AttributeError:
        logging.info('Classifier parameters unavailable')
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

    return _analyse_replacement_ranks_and_sims(df), replacement_scores


def _print_counts_data(train_counts, title):
    logging.info('----------------------')
    logging.info('| %s time statistics:' % title)
    for field in train_counts[0]._fields:
        logging.info('| %s: mean %f, std %f', field,
                     np.mean([x._asdict()[field] for x in train_counts]),
                     np.std([x._asdict()[field] for x in train_counts]))
    logging.info('----------------------')


def get_data(replacement_scores):
    x = []
    y = []
    thickness = []
    for (orig_value, repl_value), repl_count in replacement_scores.iteritems():
        y.append(repl_value)
        x.append(orig_value)
        thickness.append(repl_count)
    return x, y, thickness


def plot_dots(replacement_scores):
    x, y, thickness = get_data(replacement_scores)
    plt.scatter(x, y, np.array(thickness) / 2.0)
    plt.hlines(0, min(x), max(x))
    plt.vlines(0, min(y), max(y))
    plt.xlabel('Class association of decode-time feature')
    plt.ylabel('Class association of replacements')


def round_scores(scores):
    s = [(int(a), int(b), c) for ((a, b), c) in scores.items()]
    s = sorted(s, key=itemgetter(0, 1))
    rounded_scores1 = {}
    for key, group in groupby(s, itemgetter(0, 1)):
        rounded_scores1[key] = sum(x[2] for x in group)
    return rounded_scores1


def plot_regression_line(x, y, z):
    coef = np.polyfit(x, y, 1, w=z)
    xi = np.arange(min(x), max(x))
    line = coef[0] * xi + coef[1]
    plt.plot(xi, line, 'r-')
    plt.title('y=%.2fx+%.2f' % (coef[0], coef[1]))


def do_work(subexp='exp1-10', folds=25):
    logging.basicConfig(level=logging.INFO,
                        filename='figures/stats-%s-counts.txt' % subexp,
                        filemode='w',
                        format="%(asctime)s\t%(module)s.%(funcName)s ""(line %(lineno)d)\t%(levelname)s : %(message)s")

    train_counts, decode_counts = [], []
    basic_stats, replacement_scores = [], []

    for cv_fold in range(folds):
        logging.info('Doing fold %s', cv_fold)
        train_counts.append(_train_time_counts('stats/stats-%s-cv%d-tr.tc.csv' % (subexp, cv_fold)))

        decode_counts.append(_decode_time_counts('stats/stats-%s-cv%d-ev.tc.csv' % (subexp, cv_fold)))
        a, b = _analyse_replacements('stats/stats-%s-cv%d-ev.par.csv' % (subexp, cv_fold),
                                     'stats/stats-%s-cv%d-ev.pkl' % (subexp, cv_fold))
        basic_stats.append(a)
        replacement_scores.append(b)

    # COLLATE AND AVERAGE STATS OVER CROSSVALIDATION
    histogram_from_list(list(chain.from_iterable(x.rank for x in basic_stats)),
                        'figures/stats-%s-repl-ranks.png' % subexp)
    histogram_from_list(list(chain.from_iterable(x.sim for x in basic_stats)),
                        'figures/stats-%s-repl-sims.png' % subexp)

    _print_counts_data(train_counts, 'Train')
    _print_counts_data(decode_counts, 'Decode')

    replacement_scores = reduce(add, (Counter(x) for x in replacement_scores))

    # sometimes there may not be any IV-IT features at decode time
    if replacement_scores:
        # dump to disk so I can experiment with these counts later
        with open('stats/%s-scores.pkl' % subexp, 'w') as outf:
            pickle.dump(replacement_scores, outf)

        plt.figure()
        plot_regression_line(*get_data(replacement_scores))
        # round data in plot to make it more appealing
        plot_dots(round_scores(replacement_scores))

        plt.savefig('figures/stats-%s-NB-scores.png' % subexp, format='png')


if __name__ == '__main__':
    # do_work(subexp='exp0-0', folds=2)
    do_work(subexp='exp1-10', folds=2)
    # do_work(subexp='exp2-10', folds=10)
    do_work(subexp='exp3-10', folds=2)

