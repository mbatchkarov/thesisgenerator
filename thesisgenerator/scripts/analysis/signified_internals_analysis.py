import argparse
import os
import shelve
import sys
from discoutils.misc import ContainsEverything

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from discoutils.thesaurus_loader import Thesaurus
from sklearn.metrics.pairwise import cosine_similarity
from thesisgenerator.utils.data_utils import load_and_shelve_thesaurus
from thesisgenerator.scripts.analysis.plot import *
from thesisgenerator.scripts.analysis.utils import *
from thesisgenerator.utils.conf_file_utils import parse_config_file
from thesisgenerator.utils.misc import get_susx_mysql_conn
from thesisgenerator.plugins.stats import sum_up_token_counts
from collections import Counter
from itertools import chain, combinations
import logging
import random
import platform
import numpy as np
from scipy.stats import chisquare
from operator import add, itemgetter
from discoutils.tokens import DocumentFeature
from joblib import Parallel, delayed
import pandas as pd
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
                 classificational_and_dist_sims):
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
        self.classificational_and_dist_sims = classificational_and_dist_sims


######################################################
# BASIC (AND FAST) ANALYSIS FUNCTIONS
######################################################
def train_time_counts(fname):
    # BASIC STATISTICS AT TRAINING TIME
    df = sum_up_token_counts(fname)
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


def analyse_replacement_ranks_and_sims(df, thes):
    # BASIC STATISTICS ABOUT REPLACEMENTS (RANK IN NEIGHBOURS LIST AND SIM OT ORIGINAL)
    res_sims, res_ranks = [], []
    for i in range(1, 4):
        replacements = df['replacement%d' % i]
        sims = df['replacement%d_sim' % i]
        counts = df['count']
        for original, replacement, sim, count in zip(df.index, replacements, sims, counts):
            if sim > 0:  # filter out NaN-s
                res_sims.extend([sim] * count)
                # normally all original-s and their replacements will be in the provided thesaurus, i.e. a replacement
                # event will only occur for IV-IT features
                # For some baselines this is not the case:
                # 1. Signifier baseline does not do any replacements. The paraphrases csv file will be empty and
                # get_replacements_df will fail
                # 2. The random-neighbour replacement may result in replacements for IV-OOT, which in turn causes
                # a KeyError below
                # Most of the analysis done by this script doesn't make sense in these special cases so it's OK to fail
                rank = [x[0] for x in thes[original]].index(replacement)
                res_ranks.extend([rank] * count)

    return ReplacementsResult(res_sims, res_ranks)


def get_replacements_df(paraphrases_file):
    df = pd.read_csv(paraphrases_file, sep=', ')
    counts = df.groupby('feature').count()
    df = df.drop_duplicates()
    df.set_index('feature', inplace=True)
    df['count'] = counts.available_replacements
    return df


def get_log_odds(feature_log_probas, inv_voc):
    # ANALYSE CLASS-CONDITIONAL PROBABILITY OF REPLACEMENTS
    # this only works for binary classifiers and IV-IT features
    ratio = feature_log_probas[:, 0] - feature_log_probas[:, 1]  # P(f|c0) / P(f|c1) in log space
    # value>>0  mean strong association with class A, <<0 means the opposite.
    # I sometimes call this "class pull". This is only reliable for features the occur frequently
    # in the training set
    return {feature: ratio[index] for index, feature in inv_voc.items()}


def _get_neighbours(feature, thes, vocabulary, k):
    # this is a copy of the first bit of BaseFeatureHandler's _paraphrase method
    neighbours = [(neighbour, sim) for neighbour, sim in thes[feature] if neighbour in vocabulary]
    return neighbours[:k]


def analyse_replacements_class_pull(scores, full_voc, thes):
    it_iv_replacement_scores = defaultdict(int)
    no_IV_replacements, has_IV_replacements, good_neighbour_count, = 0, 0, 0

    for doc_feat, orig_score in scores.iteritems():  # this contains all good features in the training set (=IV, IT
        #  features with a "high" frequency). Their neighbours may or may not be contained in the given thesaurus.
        neighbours = _get_neighbours(doc_feat, thes, full_voc, 3)
        if neighbours:
            has_IV_replacements += 1
        else:
            no_IV_replacements += 1

        has_good_neighbours = False
        for i, (repl_feat, repl_sim) in enumerate(neighbours):
            if repl_feat not in scores:
                continue

            has_good_neighbours = True
            # using doubles as keys, rounding needed
            it_iv_replacement_scores[(round(orig_score, 2), round(scores[repl_feat], 2))] += 1
        if has_good_neighbours:
            good_neighbour_count += 1

    logging.info('%d/%d reliable features have any IV thes neighbours, and %d/%d dont.',
                 has_IV_replacements,
                 len(scores),
                 no_IV_replacements,
                 len(scores))
    logging.info('%d/%d reliable features have reliable neighbours', good_neighbour_count, has_IV_replacements)
    if len(it_iv_replacement_scores) < 2:
        raise ValueError('Too little data points to scatter. Need at least 2, got %d' % len(it_iv_replacement_scores))
    return it_iv_replacement_scores


#####################################################################
# FUNCTIONS THAT DO MORE ADVANCED ANALYSIS
#####################################################################

def _print_scores_of_feature_and_replacements(features, scores, counts, thes, voc):
    for feature in features:
        if feature not in scores:
            continue
        replacements = [(repl, round(scores.get(repl, -1), 2))
                        for repl, sim in _get_neighbours(feature, thes, voc, 3)
                        if repl in scores]
        if replacements:
            logging.info(' | %s (score=%2.2f, count=%d) -> %r', feature,
                         scores[feature],
                         counts[feature],
                         replacements)


def qualitative_replacement_study(scores, counts, thes, full_voc):
    logging.info('\nQualitative study of replacements in fold 0:')

    logging.info('  ---------------------------')
    logging.info(' | Most informative reliable features and their reliable replacements')
    sorted_feats_with_scores = sorted(list(scores.items()), key=itemgetter(1))
    feats_sorted_by_score = [i for i, _ in sorted_feats_with_scores]
    _print_scores_of_feature_and_replacements(feats_sorted_by_score[:10] + feats_sorted_by_score[-10:],
                                              scores, counts, thes, full_voc)
    logging.info('  ---------------------------')

    logging.info('  ---------------------------')
    logging.info(' | Most frequent reliable features and their reliable replacements')
    most_common = [x[0] for x in sorted(list(counts.items()), key=itemgetter(1), reverse=True)
                   if x[0] in thes and x[0] in scores.keys()]
    _print_scores_of_feature_and_replacements(most_common[:10], scores, counts, thes, full_voc)
    logging.info('  ---------------------------')

    logging.info('  ---------------------------')
    logging.info(' | A random sample of reliable features and their reliable replacements')
    _print_scores_of_feature_and_replacements(random.sample(feats_sorted_by_score, min(300, len(scores))),
                                              scores, counts, thes, full_voc)
    logging.info('  ---------------------------')


def correlate_similarities(classificational_vectors, inv_voc, thes):
    """
    To what extent to classification and distributional vectors agree? Calculate and correlate
    cos(a1,b1) and cos(a2,b2) for each a,b in IV IT features, where a1,b1 are distributional vectors and
    a2,b2 are classificational ones. This is slow as a thesaurus file needs to be loaded from disk.
    """

    # build a thesaurus out of the reliable classificational vectors
    cl_thes = cosine_similarity(classificational_vectors)
    logging.info('Correlating dist and clf sim of %d features that are both IV and IT', len(inv_voc))

    result = dict()
    for i, j in combinations(inv_voc.keys(), 2):
        first = inv_voc[i]
        second = inv_voc[j]
        dist_sim, class_sim = 0, 0
        # when first and second are not neighbours in the thesaurus set their sim to 0
        # todo not sure if this is optimal
        dist_neighbours = thes.get(first, [])
        for neigh, sim in dist_neighbours:
            if neigh == second:
                dist_sim = sim

        class_sim = cl_thes[i, j]
        dist_sim = dist_sim
        result[tuple(sorted([first, second]))] = (class_sim, dist_sim)
    return result


def _test_replacement_match(paraphrases_df, thes, voc):
    '''
    Make sure the replacements the classifier really makes are decode time are the ones we are using here.
    This is needed because we don't directly read the log of replacements that were made, but instead compute
    on the fly what the classifier would have done. This is needed because some of the train-time features never
    occur in the test set and are thus never replaced, so they are not in the log. We are still interested in these
    though as they provided useful information.

    :param paraphrases_df: the replacements that were actually made
    '''

    logging.info('Checking inferred and actual replacements for %d features match', len(paraphrases_df.index))
    for feature in paraphrases_df.index:  # for all features that were actually replaced
        # these are the replacements we think the classifier would have made
        inferred_replacements = [repl for repl, sim in _get_neighbours(feature, thes, voc, 3)]
        # these are the logged replacement that did take place
        recorded_replacements = []
        for i in range(1, 4):
            r = paraphrases_df['replacement%d' % i][feature]
            if r > 0:  # remove NaN
                recorded_replacements.append(r)
        assert recorded_replacements == inferred_replacements


def get_stats_for_a_single_fold(params, exp, subexp, cv_fold, thes_shelf):
    logging.info('Doing fold %d', cv_fold)
    name = 'exp%d-%d' % (exp, subexp)
    class_pulls, llr_of_good_features, it_oov_class_pulls, basic_para_stats, \
    tr_counts, ev_counts, class_and_dist_sims = [None] * 7

    if params.counts:
        logging.info('Counting')
        tr_counts = train_time_counts('statistics/stats-%s-cv%d-tr.tc.csv' % (name, cv_fold))
        ev_counts = decode_time_counts('statistics/stats-%s-cv%d-ev.tc.csv' % (name, cv_fold))

    thes = Thesaurus.from_shelf_readonly(thes_shelf)

    logging.info('Classificational vectors')
    pkl_path = 'statistics/stats-%s-cv%d-ev.MultinomialNB.pkl' % (name, cv_fold)
    all_clf_vect, full_inv_voc, all_feature_counts = load_classificational_vectors(pkl_path)
    good_clf_vect, good_inv_voc = get_good_vectors(all_clf_vect, all_feature_counts, params.min_freq, full_inv_voc,
                                                   thes)

    all_feature_counts = {v.tokens_as_str(): all_feature_counts[k] for k, v in full_inv_voc.items()}
    good_inv_voc = {k: v.tokens_as_str() for k, v in good_inv_voc.iteritems()}
    full_inv_voc = {k: v.tokens_as_str() for k, v in full_inv_voc.iteritems()}
    full_voc = set(full_inv_voc.values())

    paraphrases_file = 'statistics/stats-%s-cv%d-ev.par.csv' % (name, cv_fold)
    paraphrases_df = get_replacements_df(paraphrases_file)
    # sanity check
    _test_replacement_match(paraphrases_df, thes, full_voc)
    if params.basic_repl:
        logging.info('Loading paraphrases from disk')
        logging.info('Basic replacement stats')
        basic_para_stats = analyse_replacement_ranks_and_sims(paraphrases_df, thes)

    if params.class_pull:
        logging.info('Class pull')
        log_odds = get_log_odds(good_clf_vect, good_inv_voc)
        llr_of_good_features = analyse_replacements_class_pull(log_odds, full_voc, thes)

    if params.sim_corr:
        logging.info('Sim correlation')
        class_and_dist_sims = correlate_similarities(good_clf_vect, good_inv_voc, thes)
    if cv_fold == 0 and params.qualitative:
        logging.info('Qualitative study')
        qualitative_replacement_study(log_odds, all_feature_counts, thes, full_voc)

    return StatsOverSingleFold(tr_counts, ev_counts, basic_para_stats, llr_of_good_features,
                               it_oov_class_pulls, class_and_dist_sims)


def replacement_scores_contingency_matrix(x_scores, y_scores, weights, thresh=1):
    x = np.array(x_scores)
    y = np.array(y_scores)
    pospos = sum((x > thresh) & (y > thresh))
    posneg = sum((x > thresh) & (y < -thresh))
    negpos = sum((x < -thresh) & (y > thresh))
    negneg = sum((x < -thresh) & (y < -thresh))

    observed = [pospos, negneg, posneg, negpos]
    expected = [(pospos + negneg) / 2., (pospos + negneg) / 2., 0, 0]
    logging.info('Results of unweighted chi-square test of replacement-score contingency table: %r',
                 chisquare(observed, expected))
    logging.info('%d/%d data points have a high enough log odds score. '
                 '%d/%d data points are in the wrong quadrant.', sum(observed), len(x),
                 posneg + negpos, sum(observed))
    logging.info('Breakdown by category: pospos %d, posneg %d, negpos %d, negneg %d', pospos, posneg, negpos, negneg)


def do_work(params, exp, subexp, folds=20, workers=4, cursor=None):
    logging.info('\n\n\n\n---------------------------------------------------')
    name = 'exp%d-%d' % (exp, subexp)
    logging.info('DOING EXPERIMENT %s', name)

    plt.figure(figsize=(11, 8), dpi=300)  # for A4 print
    plt.matplotlib.rcParams.update({'font.size': 8})

    conf, configspec_file = parse_config_file('conf/exp{0}/exp{0}_base.conf'.format(exp))

    logging.info('Preparing thesaurus')
    filename = load_and_shelve_thesaurus(conf['vector_sources']['unigram_paths'],
                                         conf['vector_sources']['sim_threshold'],
                                         conf['vector_sources']['include_self'],
                                         conf['vector_sources']['allow_lexical_overlap'],
                                         conf['vector_sources']['max_neighbours'],
                                         ContainsEverything())

    all_data = Parallel(n_jobs=workers)(
        delayed(get_stats_for_a_single_fold)(params, exp, subexp, cv_fold, filename) for cv_fold in range(folds))

    logging.info('Finished all CV folds, collating')
    # COLLATE AND AVERAGE STATS OVER CROSSVALIDATION, THEN DISPLAY
    try:
        histogram_from_list(list(chain.from_iterable(x.paraphrase_stats.rank for x in all_data)),
                            1, 'Replacement ranks')
        histogram_from_list(list(chain.from_iterable(x.paraphrase_stats.sim for x in all_data)),
                            2, 'Replacement similarities')
    except AttributeError:
        pass  # didn't collect these stats, so no rank/sim attribute

    print_counts_data([x.train_counts for x in all_data], 'Train')
    print_counts_data([x.decode_counts for x in all_data], 'Decode')

    it_iv_replacement_scores = reduce(add, (Counter(x.it_iv_class_pull) for x in all_data))
    it_oov_replacement_scores = reduce(add, (Counter(x.it_oov_class_pull) for x in all_data))

    logging.info('LLR of good features %r', it_iv_replacement_scores)

    # sometimes there may not be any IV-IT features at decode time
    if it_iv_replacement_scores:
        if it_oov_replacement_scores:
            keys, values = [], []
            for k, v in it_oov_replacement_scores.items():
                keys.append(k)
                values.append(v)
            histogram_from_list(keys, 3, 'IT-OOV replacements- class associations', weights=values)

        plt.subplot(2, 3, 4)
        x, y, weights = class_pull_results_as_list(it_iv_replacement_scores)
        coef, r2, r2adj = plot_regression_line(x, y, weights)
        logging.info('R-squared of class-pull plot: %f', r2)
        # Data currently rounded to 2 significant digits. Round to nearest int to make plot less cluttered
        myrange = plot_dots(*class_pull_results_as_list(round_class_pull_to_given_precision(it_iv_replacement_scores)))
        plt.title('y=%.2fx%+.2f; r2=%.2f(%.2f); w=%s--%s' % (coef[0], coef[1], r2, r2adj, myrange[0], myrange[1]))
        logging.info('Sum-of-squares error compared to perfect diagonal = %f',
                     sum_of_squares_score_diagonal_line(x, y, weights))

        replacement_scores_contingency_matrix(x, y, weights)

    if params.sim_corr:
        class_sims = defaultdict(list)
        dist_sims = defaultdict(list)
        for x in all_data:
            for k, v in x.classificational_and_dist_sims.items():
                class_sims[k].append(v[0])
                dist_sims[k].append(v[1])

        class_sims = [np.mean(class_sims[k]) for k in sorted(class_sims.keys())]
        dist_sims = [np.mean(dist_sims[k]) for k in sorted(dist_sims.keys())]
        if class_sims and dist_sims:
            plt.subplot(2, 3, 5)
            plt.scatter(class_sims, dist_sims)
            pearson = pearsonr(class_sims, dist_sims)
            spearman = spearmanr(class_sims, dist_sims)
            _, r2, _ = plot_regression_line(class_sims, dist_sims)

            logging.info('Peason: %r', pearson)
            logging.info('Spearman: %r', spearman)
            logging.info('R-squared of sim correlation: %r', r2)
            plt.title('Pears=%.2f,Spear=%.2f, len=%d' % (pearson[0], spearman[0], len(dist_sims)))
            plt.xlabel('class_sim(a,b)')
            plt.ylabel('dist_sim(a,b)')

            x1, y1 = [], []
            for x, y in zip(class_sims, dist_sims):
                if y > 0:
                    x1.append(x)
                    y1.append(y)

            _, r2, _ = plot_regression_line(x1, y1)
            logging.info('Peason without zeroes (%d data points left): %r', len(x1), pearsonr(x1, y1))
            logging.info('Spearman without zeroes: %r', spearmanr(x1, y1))
            logging.info('R-squared of sim correlation without zeroes: %r', r2)

    if cursor:
        plt.subplot(2, 3, 6)
        s = get_experiment_info_string(cursor, exp, name)
        plt.annotate(s, (0.05, 0.5), textcoords='axes fraction')

    plt.tight_layout()
    plt.savefig('figures/stats-%s.png' % name, format='png')
    logging.info('Done all analysis')


def get_cmd_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--qualitative', action='store_true', default=False)
    parser.add_argument('--counts', action='store_true', default=False)
    parser.add_argument('--basic-repl', action='store_true', default=False)
    parser.add_argument('--class-pull', action='store_true', default=False)
    parser.add_argument('--sim-corr', action='store_true', default=False)
    parser.add_argument('--info', action='store_true', default=False)

    parser.add_argument('--experiment', type=int, default=-1)
    parser.add_argument('--min-freq', type=int, default=0)

    parser.add_argument('--all', action='store_true', default=False)
    return parser


if __name__ == '__main__':
    print os.getpid()
    parameters = get_cmd_parser().parse_args()
    logging.basicConfig(level=logging.INFO,
                        filename='figures/stats_output%d.txt' % parameters.experiment,
                        filemode='w',
                        format="%(asctime)s %(levelname)s:\t%(message)s")
    logging.getLogger().addHandler(logging.StreamHandler())

    if parameters.qualitative:
        parameters.class_pull = True  # data from class_pull stage needed for qualitative study
    if parameters.all:
        for k, v in parameters._get_kwargs():
            if v == False and isinstance(v, bool):
                parameters.__dict__[k] = True

    logging.info(parameters)

    if parameters.info:
        conn = get_susx_mysql_conn()
        c = conn.cursor() if conn else None
    else:
        c = None

    if parameters.experiment < 0:
        # do all experiments in order, all CV folds in parallel
        logging.info('Analysing all experiments')
        for i in chain(range(1, 97)):
            do_work(parameters, i, 0, folds=20, workers=20, cursor=c)
    else:
        # do just one experiment, with minimal concurrency
        logging.info('Analysing just one experiment: %d', parameters.experiment)
        hostname = platform.node()
        if not ('apollo' in hostname or 'node' in hostname):
            logging.info('RUNNING DEVELOPMENT VERSION')
            do_work(parameters, 0, 0, folds=2, workers=1, cursor=None)
        else:
            # production version
            do_work(parameters, parameters.experiment, 0, folds=20, workers=1, cursor=c)

