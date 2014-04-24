import os
import shelve
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

import matplotlib

matplotlib.use('Agg')  # so that matplotlib can run on headless machines

from discoutils.thesaurus_loader import Thesaurus
from sklearn.metrics.pairwise import cosine_similarity
from thesisgenerator.utils.conf_file_utils import parse_config_file
from thesisgenerator.utils.misc import get_susx_mysql_conn
from collections import Counter
import cPickle as pickle
from itertools import chain, groupby, combinations
import logging
from operator import add, itemgetter
from discoutils.tokens import DocumentFeature
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import pandas as pd
from thesisgenerator.plugins.stats import sum_up_token_counts
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr, pearsonr


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


def _load_classificational_vectors(pickle_file):
    with open(pickle_file) as infile:
        stats = pickle.load(infile)
    try:
        flp = stats.nb_feature_log_prob
        inv_voc = stats.nb_inv_voc
        return flp, inv_voc
    except AttributeError:
        logging.info('Classifier parameters unavailable')
        return None, None


def _analyse_replacements(paraphrases_file, flp, inv_voc):
    df = pd.read_csv(paraphrases_file, sep=', ')
    counts = df.groupby('feature').count().feature
    assert counts.sum() == df.shape[0]  # no missing rows
    df = df.drop_duplicates()
    df.set_index('feature', inplace=True)
    df['count'] = counts

    logging.info('%d/%d IV IT tokens have no replacements', sum(df['available_replacements'] == 0), len(df))

    #####################################################################
    # ANALYSE CLASS-CONDITIONAL PROBABILITY OF REPLACEMENTS
    #####################################################################

    # this only works for binary classifiers and IV-IT features
    ratio = flp[0, :] - flp[1, :]  # P(f|c0) / P(f|c1) in log space
    # high positive value mean strong association with class 0, very negative means the opposite.
    # This is called "class pull" below
    scores = {feature: ratio[index] for index, feature in inv_voc.items()}

    it_iv_replacement_scores = defaultdict(int)
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
        else:
            # this decode-time feature is IT, but OOV => we don't know it class conditional probs.
            # at least we know the class-cond probability of its replacements (because they must be IV)
            for replacement, repl_sim in get_replacements(df, f):
                if repl_sim > 0:
                    repl_score = repl_sim * scores[DocumentFeature.from_string(replacement)]
                    it_oov_replacement_scores[round(repl_score, 2)] += repl_count

    return scores, df, _analyse_replacement_ranks_and_sims(df), it_iv_replacement_scores, it_oov_replacement_scores


def _print_counts_data(train_counts, title):
    logging.info('----------------------')
    logging.info('| %s time statistics:' % title)
    for field in train_counts[0].__dict__:
        logging.info('| %s: mean %2.1f, std %2.1f', field,
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


def qualitative_replacement_study(scores, inv_voc, flp, df):
    def print_scores_of_feature_and_replacements(features, scores, counts):
        for feature in features:
            replacements = []
            for i in range(1, 4):
                r = df.ix[feature]['replacement%d' % i]
                if r > 0:  # filter out NaN-s
                    replacements.append(r)
            replacements = [(f, round(scores[f], 2)) for f in replacements]
            logging.info(' | %s (score=%2.2f, count=%d) -> %r', feature, scores[feature], counts[feature], replacements)

    logging.info('\nQualitative study of replacements in fold 0:')

    scores = {k.tokens_as_str(): v for k, v in scores.items()}
    counts = dict(df['count'])

    logging.info('  ---------------------------')
    logging.info(' | Most informative features and their replacements')
    sorted_scores = sorted(list(scores.items()), key=itemgetter(1))
    iv_it_features = [i for i, _ in sorted_scores if i in df.index]
    print_scores_of_feature_and_replacements(iv_it_features[:10] + iv_it_features[-10:], scores, counts)
    logging.info('  ---------------------------')

    logging.info('  ---------------------------')
    logging.info(' | Most frequent features and their replacements')
    most_common = [x[0] for x in sorted(list(counts.items()), key=itemgetter(1), reverse=True)
                   if x[0] in df.index and x[0] in scores.keys()]
    print_scores_of_feature_and_replacements(most_common[:10], scores, counts)
    logging.info('  ---------------------------')


def correlate_similarities(all_classificational_vectors, inv_voc, iv_it_terms, thes_shelf):
    '''
    To what extent to classification and distributional vectors agree? Calculate and correlate
    cos(a1,b1) and cos(a2,b2) for each a,b in IV IT features, where a1,b1 are distributional vectors and
    a2,b2 are classificational ones
    :param all_classificational_vectors:
    :param inv_voc:
    :param iv_it_terms:
    :param thes_file:
    :return:
    '''
    selected_rows = [row for row, feature in inv_voc.items() if feature in iv_it_terms]
    classificational_vectors = all_classificational_vectors[selected_rows, :]
    cl_thes = cosine_similarity(classificational_vectors)
    new_voc = {inv_voc[row]: i for i, row in enumerate(selected_rows)}

    d = shelve.open(thes_shelf, flag='r')  # read only
    thes = Thesaurus(d)

    dist_sims, class_sims = [], []
    for first, second in combinations(iv_it_terms, 2):
        dist_sim, class_sim = 0, 0
        # when first and second are not neighbours in the thesaurus set their sim to 0
        # todo not sure if this is optimal
        dist_neighbours = thes.get(first, [])
        for neigh, sim in dist_neighbours:
            if neigh == second:
                dist_sim = sim

        class_sims.append(cl_thes[new_voc[first], new_voc[second]])
        dist_sims.append(dist_sim)
    if len(class_sims) != len(dist_sims):
        raise ValueError
    return class_sims, dist_sims


def extract_stats_over_cv(exp, subexp, cv_fold, thes_shelf):
    name = 'exp%d-%d' % (exp, subexp)
    tr_counts = _train_time_counts('statistics/stats-%s-cv%d-tr.tc.csv' % (name, cv_fold))
    ev_counts = _decode_time_counts('statistics/stats-%s-cv%d-ev.tc.csv' % (name, cv_fold))
    flp, inv_voc = _load_classificational_vectors('statistics/stats-%s-cv%d-ev.pkl' % (name, cv_fold))

    classificational_scores, paraphrase_df, para_ranks_sims, it_iv_replacement_scores, it_oov_replacement_scores = \
        _analyse_replacements('statistics/stats-%s-cv%d-ev.par.csv' % (name, cv_fold), flp, inv_voc)
    tmp_voc = {k: v.tokens_as_str() for k, v in inv_voc.items()}

    (class_sims, dist_sims) = correlate_similarities(flp.T, tmp_voc,
                                                     [x for x in tmp_voc.values() if x in paraphrase_df.index],
                                                     thes_shelf)
    if cv_fold == 0:
        qualitative_replacement_study(classificational_scores, inv_voc, flp, paraphrase_df)
    return tr_counts, ev_counts, para_ranks_sims, it_iv_replacement_scores,\
           it_oov_replacement_scores, (class_sims, dist_sims)


def do_work(exp, subexp, folds=25, workers=4, cursor=None):
    logging.info('---------------------------------------------------')
    name = 'exp%d-%d' % (exp, subexp)
    logging.info('Doing experiment %s', name)

    plt.figure(figsize=(11, 8), dpi=300)  # for A4 print
    plt.matplotlib.rcParams.update({'font.size': 8})

    conf, configspec_file = parse_config_file('conf/exp{0}/exp{0}_base.conf'.format(exp))
    thes_file = conf['vector_sources']['unigram_paths'][0]
    filename = 'shelf%d' % hash(tuple([thes_file]))
    if not os.path.exists(filename):
        thes = Thesaurus.from_tsv([thes_file])
        thes.to_shelf(filename)
    res = Parallel(n_jobs=workers)(delayed(extract_stats_over_cv)(exp, subexp, cv_fold, filename)
                                   for cv_fold in range(folds))
    # res is a list of
    # tr_counts, ev_counts, para_ranks_sims, it_iv_replacement_scores, it_oov_replacement_scores, (class_sims, dist_sims)
    train_counts = [x[0] for x in res]
    decode_counts = [x[1] for x in res]
    basic_repl_stats = [x[2] for x in res]
    it_iv_replacement_scores = [x[3] for x in res]
    it_oov_replacement_scores = [x[4] for x in res]
    class_sims = [x[5][0] for x in res]
    dist_sims = [x[5][1] for x in res]

    class_sims = list(chain.from_iterable(class_sims))
    dist_sims = list(chain.from_iterable(dist_sims))

    # COLLATE AND AVERAGE STATS OVER CROSSVALIDATION
    histogram_from_list(list(chain.from_iterable(x.rank for x in basic_repl_stats)), 1, 'Replacement ranks')
    histogram_from_list(list(chain.from_iterable(x.sim for x in basic_repl_stats)), 2, 'Replacement similarities')

    _print_counts_data(train_counts, 'Train')
    _print_counts_data(decode_counts, 'Decode')

    it_iv_replacement_scores = reduce(add, (Counter(x) for x in it_iv_replacement_scores))
    it_oov_replacement_scores = reduce(add, (Counter(x) for x in it_oov_replacement_scores))

    keys, values = [], []
    for k, v in it_oov_replacement_scores.items():
        keys.append(k)
        values.append(v)
    histogram_from_list(keys, 3, 'IT-OOV replacements- class associations', weights=values)

    # sometimes there may not be any IV-IT features at decode time
    if it_iv_replacement_scores:
        # dump to disk so I can experiment with these counts later
        with open('statistics/%s-scores.pkl' % name, 'w') as outf:
            pickle.dump(it_iv_replacement_scores, outf)

        plt.subplot(2, 3, 4)
        coef = plot_regression_line(*get_data(it_iv_replacement_scores))
        # Data currently rounded to 2 significant digits. Round to nearest int to make plot less cluttered
        myrange = plot_dots(round_scores_to_given_precision(it_iv_replacement_scores))
        plt.title('y=%.2fx%+.2f; w=%s--%s' % (coef[0], coef[1], myrange[0], myrange[1]))

    if class_sims and dist_sims:
        plt.subplot(2, 3, 5)
        plt.scatter(class_sims, dist_sims)
        pearson = pearsonr(class_sims, dist_sims)
        spearman = spearmanr(class_sims, dist_sims)
        logging.info('Peason: %r', pearson)
        logging.info('Spearman: %r', spearman)
        plt.title('Pears=%.2f,Spear=%.2f, len=%d' % (pearson[0], spearman[0], len(dist_sims)))
        plt.xlabel('class_sim(a,b)')
        plt.ylabel('dist_sim(a,b)')

        x1, y1 = [], []
        for x, y in zip(class_sims, dist_sims):
            if y > 0:
                x1.append(x)
                y1.append(y)

        logging.info('Peason without zeroes (%d data points left): %r', len(x1), pearsonr(x1, y1))
        logging.info('Spearman without zeroes: %r', spearmanr(x1, y1))
    if cursor:
        plt.subplot(2, 3, 6)
        query = 'SELECT score_mean, score_std FROM data%d WHERE NAME="%s" AND ' \
                'metric="macroavg_f1" AND classifier="MultinomialNB"' % (exp, name)
        cursor.execute(query)
        f1 = 'F1=%.2f+-%1.2f' % cursor.fetchall()[0]

        query = "SELECT * FROM ExperimentDescriptions WHERE number=%d" % exp
        cursor.execute(query)
        settings_data = cursor.fetchone()
        desc = cursor.description
        settings = {}

        for (key, value) in zip(desc, settings_data):
            settings[key[0]] = value

        s = '\n'.join('%s:%s' % (str(k), str(v)) for k, v in settings.items())
        plt.annotate('%s: %s\n%s' % (name, s, f1), (0.05, 0.5), textcoords='axes fraction')

    plt.tight_layout()
    plt.savefig('figures/stats-%s.png' % name, format='png')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        filename='figures/stats_output.txt',
                        filemode='w',
                        format="%(message)s")
    logging.getLogger().addHandler(logging.StreamHandler())

    conn = get_susx_mysql_conn()
    c = conn.cursor() if conn else None

    # do_work(0, 0, folds=2, workers=1)
    do_work(1, 0, folds=10, workers=5)
    do_work(1, 5, folds=10, workers=5)

    # for i in range(1, 45):
    #     do_work(i, 5, folds=20, workers=20, cursor=c)
    #
    # for i in range(57, 63):
    #     do_work(i, 5, folds=20, workers=20, cursor=c)

