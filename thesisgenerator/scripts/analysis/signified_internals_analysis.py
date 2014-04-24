import argparse
import os
import shelve
import sys

from thesisgenerator.scripts.analysis.plot import *
from thesisgenerator.scripts.analysis.utils import *


sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

import matplotlib.pyplot as plt
from discoutils.thesaurus_loader import Thesaurus
from sklearn.metrics.pairwise import cosine_similarity
from thesisgenerator.utils.conf_file_utils import parse_config_file
from thesisgenerator.utils.misc import get_susx_mysql_conn
from collections import Counter
import cPickle as pickle
from itertools import chain, combinations
import logging
from operator import add, itemgetter
from discoutils.tokens import DocumentFeature
from joblib import Parallel, delayed
import pandas as pd
from thesisgenerator.plugins.stats import sum_up_token_counts
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr

######################################################
# CLASSES TO STORE RESULTS OF DIFFERENT ANALYSES
######################################################

class DecodeTimeCounts(object):
    def __init__(self, *args):
        self.tokens, self.types, self.iv_types, self.it_types, self.iv, self.it, \
        self.iv_it, self.iv_oot, self.oov_it, self.oov_oot = args


class TrainCount(object):
    def __init__(self, *args):
        self.tokens, self.types, self.it_tokens, self.it_types = args


class ReplacementsResult(object):
    def __init__(self, sim, rank):
        self.rank, self.sim = rank, sim


class StatsOverSingleFold(object):
    def __init__(self, train_counts, decode_counts, paraphrase_stats, it_iv_class_pull, it_oov_class_pull,
                 classificational_sims, distributional_sims):
        """

        :type train_counts: TrainCount
        :type decode_counts: DecodeTimeCounts
        :type paraphrase_stats: ReplacementsResult
        :type it_iv_class_pull: dict
        :type it_oov_class_pull: dict
        :type classificational_sims: list
        :type distributional_sims: list
        """
        self.train_counts = train_counts
        self.decode_counts = decode_counts
        self.paraphrase_stats = paraphrase_stats
        self.it_iv_class_pull = it_iv_class_pull
        self.it_oov_class_pull = it_oov_class_pull
        self.classificational_sims = classificational_sims
        self.distributional_sims = distributional_sims


######################################################
# BASIC (AND FAST) ANALYSIS FUNCTIONS
######################################################
def train_time_counts(fname):
    # BASIC STATISTICS AT TRAINING TIME
    df = sum_up_token_counts(fname)
    vocab = set(df.index.tolist())
    df_it = df[df['IT'] > 0]
    return TrainCount(df['count'].sum(),
                      df.shape[0],
                      df_it['count'].sum(),
                      df_it.shape[0])


def decode_time_counts(fname):
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


def analyse_replacement_ranks_and_sims(df):
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
        res[statistic] = data
    return ReplacementsResult(**res)


def get_replacements_df(paraphrases_file):
    df = pd.read_csv(paraphrases_file, sep=', ')
    counts = df.groupby('feature').count().feature
    assert counts.sum() == df.shape[0]  # no missing rows
    df = df.drop_duplicates()
    df.set_index('feature', inplace=True)
    df['count'] = counts
    return df


def analyse_replacements_class_pull(df, flp, inv_voc):
    def get_replacements(df, feature):
        for i in range(1, 4):
            repl_feature = df.ix[feature]['replacement%d' % i]
            repl_sim = df.ix[feature]['replacement%d_sim' % i]
            if repl_sim > 0:
                yield repl_feature, repl_sim

    logging.info('%d/%d IV IT tokens have no replacements', sum(df['available_replacements'] == 0), len(df))

    # ANALYSE CLASS-CONDITIONAL PROBABILITY OF REPLACEMENTS
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

    return scores, it_iv_replacement_scores, it_oov_replacement_scores


#####################################################################
# FUNCTIONS THAT DO MORE ADVANCED ANALYSIS
#####################################################################

def qualitative_replacement_study(class_pulls, repl_df):
    def print_scores_of_feature_and_replacements(features, scores, counts):
        for feature in features:
            replacements = []
            for i in range(1, 4):
                r = repl_df.ix[feature]['replacement%d' % i]
                if r > 0:  # filter out NaN-s
                    replacements.append(r)
            replacements = [(f, round(scores[f], 2)) for f in replacements]
            logging.info(' | %s (score=%2.2f, count=%d) -> %r', feature,
                         scores[feature], counts[feature], replacements)

    logging.info('\nQualitative study of replacements in fold 0:')

    class_pulls = {k.tokens_as_str(): v for k, v in class_pulls.items()}
    counts = dict(repl_df['count'])

    logging.info('  ---------------------------')
    logging.info(' | Most informative features and their replacements')
    sorted_scores = sorted(list(class_pulls.items()), key=itemgetter(1))
    iv_it_features = [i for i, _ in sorted_scores if i in repl_df.index]
    print_scores_of_feature_and_replacements(iv_it_features[:10] + iv_it_features[-10:], class_pulls, counts)
    logging.info('  ---------------------------')

    logging.info('  ---------------------------')
    logging.info(' | Most frequent features and their replacements')
    most_common = [x[0] for x in sorted(list(counts.items()), key=itemgetter(1), reverse=True)
                   if x[0] in repl_df.index and x[0] in class_pulls.keys()]
    print_scores_of_feature_and_replacements(most_common[:10], class_pulls, counts)
    logging.info('  ---------------------------')


def correlate_similarities(all_classificational_vectors, inv_voc, iv_it_terms, thes_shelf):
    """
    To what extent to classification and distributional vectors agree? Calculate and correlate
    cos(a1,b1) and cos(a2,b2) for each a,b in IV IT features, where a1,b1 are distributional vectors and
    a2,b2 are classificational ones. This is slow as a thesaurus file needs to be loaded from disk.
    """
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


def extract_stats_for_a_single_fold(exp, subexp, cv_fold, thes_shelf, do_slow_bit):
    name = 'exp%d-%d' % (exp, subexp)
    tr_counts = train_time_counts('statistics/stats-%s-cv%d-tr.tc.csv' % (name, cv_fold))
    ev_counts = decode_time_counts('statistics/stats-%s-cv%d-ev.tc.csv' % (name, cv_fold))
    flp, inv_voc = load_classificational_vectors('statistics/stats-%s-cv%d-ev.pkl' % (name, cv_fold))

    paraphrases_file = 'statistics/stats-%s-cv%d-ev.par.csv' % (name, cv_fold)
    paraphrases_df = get_replacements_df(paraphrases_file)
    basic_para_stats = analyse_replacement_ranks_and_sims(paraphrases_df)
    class_pulls, it_iv_class_pulls, it_oov_class_pulls = analyse_replacements_class_pull(paraphrases_df, flp, inv_voc)

    if do_slow_bit:
        tmp_voc = {k: v.tokens_as_str() for k, v in inv_voc.items()}
        class_sims, dist_sims = correlate_similarities(flp.T, tmp_voc,
                                                       [x for x in tmp_voc.values() if x in paraphrases_df.index],
                                                       thes_shelf)
    else:
        class_sims, dist_sims = [], []

    if cv_fold == 0:
        qualitative_replacement_study(class_pulls, paraphrases_df)

    return StatsOverSingleFold(tr_counts, ev_counts, basic_para_stats, it_iv_class_pulls,
                               it_oov_class_pulls, class_sims, dist_sims)


def do_work(exp, subexp, folds=25, workers=4, cursor=None, do_slow_bit=False):
    logging.info('---------------------------------------------------')
    name = 'exp%d-%d' % (exp, subexp)
    logging.info('Doing experiment %s', name)

    plt.figure(figsize=(11, 8), dpi=300)  # for A4 print
    plt.matplotlib.rcParams.update({'font.size': 8})

    conf, configspec_file = parse_config_file('conf/exp{0}/exp{0}_base.conf'.format(exp))
    thes_file = conf['vector_sources']['unigram_paths'][0]
    filename = 'shelf%d' % hash(tuple([thes_file]))
    if do_slow_bit:
        if not os.path.exists(filename):
            thes = Thesaurus.from_tsv([thes_file])
            thes.to_shelf(filename)

    all_data = Parallel(n_jobs=workers)(
        delayed(extract_stats_for_a_single_fold)(exp, subexp, cv_fold, filename, do_slow_bit) for cv_fold in range(folds))

    # COLLATE AND AVERAGE STATS OVER CROSSVALIDATION, THEN DISPLAY
    histogram_from_list(list(chain.from_iterable(x.paraphrase_stats.rank for x in all_data)),
                        1, 'Replacement ranks')
    histogram_from_list(list(chain.from_iterable(x.paraphrase_stats.sim for x in all_data)),
                        2, 'Replacement similarities')

    print_counts_data([x.train_counts for x in all_data], 'Train')
    print_counts_data([x.decode_counts for x in all_data], 'Decode')

    it_iv_replacement_scores = reduce(add, (Counter(x.it_iv_class_pull) for x in all_data))
    it_oov_replacement_scores = reduce(add, (Counter(x.it_oov_class_pull) for x in all_data))

    # sometimes there may not be any IV-IT features at decode time
    if it_iv_replacement_scores and it_oov_replacement_scores:
        keys, values = [], []
        for k, v in it_oov_replacement_scores.items():
            keys.append(k)
            values.append(v)
        histogram_from_list(keys, 3, 'IT-OOV replacements- class associations', weights=values)
        # dump to disk so I can experiment with these counts later
        with open('statistics/%s-scores.pkl' % name, 'w') as outf:
            pickle.dump(it_iv_replacement_scores, outf)

        plt.subplot(2, 3, 4)
        coef = plot_regression_line(*class_pull_results_as_list(it_iv_replacement_scores))  # todo r^2 statistic
        # Data currently rounded to 2 significant digits. Round to nearest int to make plot less cluttered
        myrange = plot_dots(round_class_pull_to_given_precision(it_iv_replacement_scores))
        plt.title('y=%.2fx%+.2f; w=%s--%s' % (coef[0], coef[1], myrange[0], myrange[1]))

    class_sims = list(chain.from_iterable(x.classificational_sims for x in all_data))
    dist_sims = list(chain.from_iterable(x.distributional_sims for x in all_data))
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
        s = get_experiment_info_string(cursor, exp, name)
        plt.annotate(s, (0.05, 0.5), textcoords='axes fraction')

    plt.tight_layout()
    plt.savefig('figures/stats-%s.png' % name, format='png')


def get_cmd_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--slow', action='store_true', default=False,
                        help='If set, will also do the very slow piece of analysis')
    return parser


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        filename='figures/stats_output.txt',
                        filemode='w',
                        format="%(message)s")
    logging.getLogger().addHandler(logging.StreamHandler())

    parameters = get_cmd_parser().parse_args()
    logging.info(parameters)

    conn = get_susx_mysql_conn()
    c = conn.cursor() if conn else None

    do_work(0, 0, folds=2, workers=1)
    # do_work(1, 0, folds=10, workers=5, cursor=c)
    # do_work(1, 5, folds=10, workers=5)

    # for i in range(1, 45):
    #     do_work(i, 5, folds=20, workers=20, cursor=c, do_slow_bit=parameters.slow)
    #
    # for i in range(57, 63):
    #     do_work(i, 5, folds=20, workers=20, cursor=c, do_slow_bit=parameters.slow)

