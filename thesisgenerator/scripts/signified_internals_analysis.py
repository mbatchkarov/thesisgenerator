from collections import Counter
import cPickle as pickle
from itertools import chain, groupby
import logging
from operator import add, itemgetter
from discoutils.tokens import DocumentFeature
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import pandas as pd
from thesisgenerator.plugins.stats import sum_up_token_counts
from collections import defaultdict
import numpy as np


def histogram_from_list(l, subplot, title, weights=None):
    MAX_LABEL_COUNT = 40
    plt.subplot(2, 3, subplot)
    if type(l[0]) == str:
        # numpy's histogram doesn't like strings
        s = pd.Series(Counter(l))
        s.plot(kind='bar', rot=0, title=title)
    else:
        plt.hist(l, bins=MAX_LABEL_COUNT, weights=weights)
        plt.title(title)


class TrainCount(object):
    def __init__(self, *args):
        self.tokens, self.types, self.it_tokens, self.it_types = args


def _train_time_counts(fname):
    # BASIC STATISTICS AT TRAINING TIME
    df = sum_up_token_counts(fname)
    vocab = set(df.index.tolist())
    df_it = df[df['IT'] > 0]
    return TrainCount(df['count'].sum(),
                      df.shape[0],
                      df_it['count'].sum(),
                      df_it.shape[0])


class DecodeTimeCounts(object):
    def __init__(self, *args):
        self.tokens, self.types, self.iv_types, self.it_types, self.iv, self.it, \
        self.iv_it, self.iv_oot, self.oov_it, self.oov_oot = args


def _decode_time_counts(fname):
    # BASIC STATISTICS AT DECODE TIME
    df = sum_up_token_counts(fname)
    df_it = df[df['IT'] > 0]
    df_iv = df[df['IV'] > 0]

    return DecodeTimeCounts(
        df['count'].sum(),
        df.shape[0],
        df_iv.shape[0],
        df_it.shape[0],
        df_it['count'].sum(),
        df_iv['count'].sum(),
        df[(df['IT'] > 0) & (df['IV'] > 0)]['count'].sum(),
        df[(df['IT'] > 0) & (df['IV'] == 0)]['count'].sum(),
        df[(df['IT'] == 0) & (df['IV'] > 0)]['count'].sum(),
        df[(df['IT'] == 0) & (df['IV'] == 0)]['count'].sum()
    )


class ReplacementsResult(object):
    def __init__(self, sim, rank):
        self.rank, self.sim = rank, sim


def _analyse_replacement_ranks_and_sims(df):
    # BASIC STATISTICS ABOUT REPLACEMENTS (RANK IN NEIGHBOURS LIST AND SIM OT ORIGINAL)
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
    return ReplacementsResult(**res)


def get_replacements(df, feature):
    for i in range(1, 4):
        repl_feature = df.ix[feature]['replacement%d' % i]
        repl_sim = df.ix[feature]['replacement%d_sim' % i]
        if repl_sim > 0:
            yield repl_feature, repl_sim


def _analyse_replacements(paraphrases_file, pickle_file):
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
        # high positive value mean strong association with class 0, very negative means the opposite.
        # This is called "class pull" below
        scores = {feature: ratio[index] for index, feature in stats.nb_inv_voc.items()}
    except AttributeError:
        logging.info('Classifier parameters unavailable')
        return None, None

    it_iv_replacement_scores = defaultdict(int)
    it_iv_sim_vs_replacement_ratio = defaultdict(int)
    it_oov_replacement_scores = defaultdict(int)
    for f in df.index:  # this contains all IT features, regardless of whether they were IV or IT
        orig_score = scores.get(DocumentFeature.from_string(f), None)
        repl_count = df.ix[f]['count']
        if orig_score:
            # this is an IT, IV feature
            if repl_count > 0:
                for replacement, repl_sim in get_replacements(df, f):
                    if repl_sim > 0:
                        # -1 signifies no replacement has been found
                        repl_score = repl_sim * scores[DocumentFeature.from_string(replacement)]
                        it_iv_replacement_scores[(round(orig_score, 2), round(repl_score, 2))] += repl_count
                        # ( P(f0|c0) / P(f0|c1) ) / (P(f1|c0) / P(f1|c1)) in log space. This doesn't yet express
                        # what I want by I'm keeping it anyway
                        ratio = orig_score - scores[DocumentFeature.from_string(replacement)]
                        it_iv_sim_vs_replacement_ratio[(round(repl_sim, 2), round(ratio, 2))] += repl_count
        else:
            # this decode-time feature is IT, but OOV => we don't know it class conditional probs.
            # at least we know the class-cond probability of its replacements (because they must be IV)
            for replacement, repl_sim in get_replacements(df, f):
                if repl_sim > 0:
                    it_oov_replacement_scores[round(repl_sim, 2)] += repl_count

    return _analyse_replacement_ranks_and_sims(df), it_iv_replacement_scores, it_oov_replacement_scores, \
           it_iv_sim_vs_replacement_ratio


def _print_counts_data(train_counts, title):
    logging.info('----------------------')
    logging.info('| %s time statistics:' % title)
    for field in train_counts[0].__dict__:
        logging.info('| %s: mean %f, std %f', field,
                     np.mean([getattr(x, field) for x in train_counts]),
                     np.std([getattr(x, field) for x in train_counts]))
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


def plot_dots(replacement_scores, minsize=10., maxsize=200., draw_axes=True,
              xlabel='Class association of decode-time feature',
              ylabel='Class association of replacements'):
    x, y, thickness = get_data(replacement_scores)
    z = np.array(thickness)
    range = min(z), max(z)
    if min(z) < minsize:
        z += (minsize - min(z))

    # http://stackoverflow.com/a/17029736/419338
    normalized_z = ((maxsize - minsize) * (z - min(z))) / (max(z) - min(z)) + minsize

    plt.scatter(x, y, normalized_z)
    if draw_axes:
        plt.hlines(0, min(x), max(x))
        plt.vlines(0, min(y), max(y))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return range


def round_scores_to_given_precision(scores, xprecision=0, yprecision=0):
    '''
    Rounds keys in dict to nearest integer. Dict must have the following structure
     (key1: float, key2: float): value:float

    Entries that fall into the same bin after rounding are added up, e.g.
    >>> round_scores_to_given_precision({(1.1, 2.1):3, (1.111, 2.111):3})
    {(1.0, 2.0): 6}
    >>> round_scores_to_given_precision({(1.1, 2.1):3, (1.111, 2.111):3}, 1, 1)
    {(1.1, 2.1): 6}
    '''
    s = [(round(a, xprecision), round(b, yprecision), c) for ((a, b), c) in scores.items()]
    s = sorted(s, key=itemgetter(0, 1))
    rounded_scores = {}
    for key, group in groupby(s, itemgetter(0, 1)):
        rounded_scores[key] = sum(x[2] for x in group)
    return rounded_scores


def plot_regression_line(x, y, z):
    coef = np.polyfit(x, y, 1, w=z)
    xi = np.linspace(min(x), max(x))
    line = coef[0] * xi + coef[1]
    plt.plot(xi, line, 'r-')
    return coef


def extract_stats_over_cv(subexp, cv_fold):
    logging.info('Doing fold %s', cv_fold)
    a = _train_time_counts('stats/stats-%s-cv%d-tr.tc.csv' % (subexp, cv_fold))
    b = _decode_time_counts('stats/stats-%s-cv%d-ev.tc.csv' % (subexp, cv_fold))
    c, d, f, g = _analyse_replacements('stats/stats-%s-cv%d-ev.par.csv' % (subexp, cv_fold),
                                       'stats/stats-%s-cv%d-ev.pkl' % (subexp, cv_fold))
    return a, b, c, d, f, g


def do_work(subexp, folds=25, workers=4):
    logging.info('---------------------------------------------------')
    logging.info('Doing experiment %s', subexp)
    plt.figure(figsize=(11, 8), dpi=300)  # for A4 print
    plt.matplotlib.rcParams.update({'font.size': 9})

    res = Parallel(n_jobs=workers)(delayed(extract_stats_over_cv)(subexp, cv_fold) for cv_fold in range(folds))

    train_counts = [x[0] for x in res]
    decode_counts = [x[1] for x in res]
    basic_repl_stats = [x[2] for x in res]
    it_iv_replacement_scores = [x[3] for x in res]
    it_oov_replacement_scores = [x[4] for x in res]
    it_iv_sim_vs_ratio = [x[5] for x in res]

    # COLLATE AND AVERAGE STATS OVER CROSSVALIDATION
    histogram_from_list(list(chain.from_iterable(x.rank for x in basic_repl_stats)), 1, 'Replacement ranks')
    histogram_from_list(list(chain.from_iterable(x.sim for x in basic_repl_stats)), 2, 'Replacement similarities')

    _print_counts_data(train_counts, 'Train')
    _print_counts_data(decode_counts, 'Decode')

    it_iv_replacement_scores = reduce(add, (Counter(x) for x in it_iv_replacement_scores))
    it_oov_replacement_scores = reduce(add, (Counter(x) for x in it_oov_replacement_scores))
    it_iv_sim_vs_ratio = reduce(add, (Counter(x) for x in it_iv_sim_vs_ratio))

    keys, values = [], []
    for k, v in it_oov_replacement_scores.items():
        keys.append(k)
        values.append(v)
    histogram_from_list(keys, 3, 'IT-OOV replacements- class associations', weights=values)

    # sometimes there may not be any IV-IT features at decode time
    if it_iv_replacement_scores:
        # dump to disk so I can experiment with these counts later
        with open('stats/%s-scores.pkl' % subexp, 'w') as outf:
            pickle.dump(it_iv_replacement_scores, outf)

        plt.subplot(2, 3,  4)
        coef = plot_regression_line(*get_data(it_iv_replacement_scores))
        # Data currently rounded to 2 significant digits. Round to nearest int to make plot less cluttered
        myrange = plot_dots(round_scores_to_given_precision(it_iv_replacement_scores))
        plt.title('y=%.2fx%+.2f; w=%s--%s' % (coef[0], coef[1], myrange[0], myrange[1]))

    if it_iv_sim_vs_ratio:
        plt.subplot(2, 3,  5)
        coef = plot_regression_line(*get_data(it_iv_sim_vs_ratio))
        myrange = plot_dots(round_scores_to_given_precision(it_iv_sim_vs_ratio, 1, 0),
                            xlabel='sim(a,b)',
                            ylabel='log(ratio(a))-log(ratio(b))',
                            draw_axes=False)
        plt.title('y=%.2fx%+.2f; w=%s--%s' % (coef[0], coef[1], myrange[0], myrange[1]))

    plt.tight_layout()
    plt.savefig('figures/stats-%s.png' % subexp, format='png')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        filename='figures/stats_output.txt',
                        filemode='w',
                        format="%(asctime)s\t%(module)s.%(funcName)s ""(line %(lineno)d)\t%(levelname)s : %(message)s")

    do_work(subexp='exp0-0', folds=2, workers=1)
    do_work('exp1-10', folds=4)
    do_work('exp3-10', folds=4)

