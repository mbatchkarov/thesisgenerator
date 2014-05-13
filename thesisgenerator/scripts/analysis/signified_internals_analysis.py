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


def get_class_pull(flp, frequent_inv_voc):
    # ANALYSE CLASS-CONDITIONAL PROBABILITY OF REPLACEMENTS
    # this only works for binary classifiers and IV-IT features
    ratio = flp[:, 0] - flp[:, 1]  # P(f|c0) / P(f|c1) in log space (odds ratio?)
    # value>>0  mean strong association with class A, <<0 means the opposite.
    # This is called "class pull" below. This is only reliable for features the occur frequently
    # in the training set
    reliable_scores = {feature: ratio[index] for index, feature in frequent_inv_voc.items()}
    return reliable_scores


def analyse_replacements_class_pull(reliable_scores, full_inv_voc, thes):
    def get_neighbours(feature, thes, vocabulary, k):
        # this is a copy of the first bit of BaseFeatureHandler's _paraphrase method
        neighbours = thes[feature]
        neighbours = [(neighbour, sim) for neighbour, sim in neighbours if neighbour in vocabulary]
        return neighbours[:k]


    voc = set(x.tokens_as_str() for x in full_inv_voc.values())
    it_iv_replacement_scores = defaultdict(int)
    it_oov_replacement_scores = defaultdict(int)
    not_in_thes, IT_but_no_IV_replacements, IT_has_IV_replacements = 0, 0, 0
    for doc_feat, orig_score in reliable_scores.iteritems():  # this contains all IV features with a
        # "high" frequency in the training set. These may or may not be contained in the given thesaurus.
        if doc_feat in thes:
            neighbours = get_neighbours(doc_feat, thes, voc, 3)
            if neighbours:
                # logging.info('No IV replacements')
                IT_has_IV_replacements += 1
            else:
                IT_but_no_IV_replacements += 1
            for i, (repl_feat, repl_sim) in enumerate(neighbours):
                if repl_feat not in reliable_scores:
                    # logging.info('%d Replacement not reliable', i)
                    continue
                # logging.info('Replacement score is reliable')
                # todo considers the similarity between an entry and its neighbours
                repl_score = repl_sim * reliable_scores[repl_feat]
                # using doubles as keys, rounding needed
                it_iv_replacement_scores[(round(orig_score, 2), round(repl_score, 2))] += 1
        else:
            # logging.info('Original not in thesaurus')
            not_in_thes += 1
            pass
            # # this decode-time feature is IT, but OOV => we don't know it class conditional probs.
            # # at least we know the class-cond probability of its replacements (because they must be IV)
            # for replacement_str, repl_sim in get_neighbours(doc_feat, thes, voc, 3):
            #     if replacement_str not in reliable_scores:
            #         continue
            #     repl_score = repl_sim * reliable_scores[replacement_str]
            #     it_oov_replacement_scores[round(repl_score, 2)] += 1

    logging.info('%d/%d reliable features not in thesaurus', not_in_thes, len(reliable_scores))
    logging.info('%d/%d reliable features have no IV thes neighbours, and %d do. These may not be reliable though',
                 IT_but_no_IV_replacements,
                 len(reliable_scores),
                 IT_has_IV_replacements)
    if len(it_iv_replacement_scores) < 2:
        raise ValueError('Too little data points to scatter. Need at least 2, got %d' % len(it_iv_replacement_scores))
    return it_iv_replacement_scores, it_oov_replacement_scores


#####################################################################
# FUNCTIONS THAT DO MORE ADVANCED ANALYSIS
#####################################################################

def qualitative_replacement_study(class_pulls, repl_df):
    def print_scores_of_feature_and_replacements(features, scores, counts):
        for feature in features:
            if feature not in class_pulls:
                continue
            replacements = []
            for i in range(1, 4):
                r = repl_df.ix[feature]['replacement%d' % i]
                if r > 0 and r in class_pulls:  # filter out NaN-s
                    replacements.append(r)
            replacements = [(f, round(scores[f], 2)) for f in replacements]
            logging.info(' | %s (score=%2.2f, count=%d) -> %r', feature,
                         scores[feature], counts[feature], replacements)

    logging.info('\nQualitative study of replacements in fold 0:')

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


def correlate_similarities(reliable_classificational_vectors, inv_voc, iv_it_terms, thes):
    """
    To what extent to classification and distributional vectors agree? Calculate and correlate
    cos(a1,b1) and cos(a2,b2) for each a,b in IV IT features, where a1,b1 are distributional vectors and
    a2,b2 are classificational ones. This is slow as a thesaurus file needs to be loaded from disk.
    """

    # some reliable vocabulary items were not in thesaurus, remove them
    selected_rows = [row for row, feature in inv_voc.items() if feature in iv_it_terms]
    iv_it_voc = {inv_voc[row]: idx for idx, row in enumerate(selected_rows)}
    classificational_vectors = reliable_classificational_vectors[selected_rows, :]
    # build a thesaurus out of the remaining classificational vectors
    cl_thes = cosine_similarity(classificational_vectors)
    logging.info('Correlating dist and clf sim of %d features that are both IV and IT', len(iv_it_voc))

    result = dict()
    for first, second in combinations(iv_it_voc.keys(), 2):
        dist_sim, class_sim = 0, 0
        # when first and second are not neighbours in the thesaurus set their sim to 0
        # todo not sure if this is optimal
        dist_neighbours = thes.get(first, [])
        for neigh, sim in dist_neighbours:
            if neigh == second:
                dist_sim = sim

        class_sim = cl_thes[iv_it_voc[first], iv_it_voc[second]]
        dist_sim = dist_sim
        result[tuple(sorted([first, second]))] = (class_sim, dist_sim)
    return result


def get_stats_for_a_single_fold(params, exp, subexp, cv_fold, thes_shelf):
    logging.info('Doing fold %d', cv_fold)
    name = 'exp%d-%d' % (exp, subexp)
    class_pulls, it_iv_class_pulls, it_oov_class_pulls, basic_para_stats, \
    tr_counts, ev_counts, class_and_dist_sims = [None] * 7

    if params.counts:
        logging.info('Counting')
        tr_counts = train_time_counts('statistics/stats-%s-cv%d-tr.tc.csv' % (name, cv_fold))
        ev_counts = decode_time_counts('statistics/stats-%s-cv%d-ev.tc.csv' % (name, cv_fold))

    d = shelve.open(thes_shelf, flag='r')  # read only
    thes = Thesaurus(d)

    logging.info('Classificational vectors')
    pkl_path = 'statistics/stats-%s-cv%d-ev.MultinomialNB.pkl' % (name, cv_fold)
    reliable_clf_vect, reliable_inv_voc, full_inv_voc = load_classificational_vectors(pkl_path, params.min_freq)
    logging.info('%d features with reliable clf vectors are in thesaurus',
                 sum(x.tokens_as_str() in thes for x in reliable_inv_voc.values()))

    if params.basic_repl or params.class_pull:
        logging.info('Loading paraphrases from disk')
        paraphrases_file = 'statistics/stats-%s-cv%d-ev.par.csv' % (name, cv_fold)
        paraphrases_df = get_replacements_df(paraphrases_file)
        if params.basic_repl:
            logging.info('Basic replacement stats')
            basic_para_stats = analyse_replacement_ranks_and_sims(paraphrases_df, thes)
        if params.class_pull:
            logging.info('Class pull')
            reliable_scores = get_class_pull(reliable_clf_vect, reliable_inv_voc)
            reliable_scores = {k.tokens_as_str(): v for k, v in reliable_scores.iteritems()}
            it_iv_class_pulls, it_oov_class_pulls = analyse_replacements_class_pull(reliable_scores, full_inv_voc, thes)

    if params.sim_corr:
        logging.info('Sim correlation')
        reliable_inv_voc = {k: v.tokens_as_str() for k, v in reliable_inv_voc.items()}
        class_and_dist_sims = correlate_similarities(reliable_clf_vect, reliable_inv_voc,
                                                     [x for x in reliable_inv_voc.values() if x in thes],
                                                     thes)
    if cv_fold == 0 and params.qualitative:
        logging.info('Qualitative study')
        qualitative_replacement_study(reliable_scores, paraphrases_df)

    return StatsOverSingleFold(tr_counts, ev_counts, basic_para_stats, it_iv_class_pulls,
                               it_oov_class_pulls, class_and_dist_sims)


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

    # sometimes there may not be any IV-IT features at decode time
    if it_iv_replacement_scores:
        if it_oov_replacement_scores:
            keys, values = [], []
            for k, v in it_oov_replacement_scores.items():
                keys.append(k)
                values.append(v)
            histogram_from_list(keys, 3, 'IT-OOV replacements- class associations', weights=values)

        plt.subplot(2, 3, 4)
        coef, r2, r2adj = plot_regression_line(*class_pull_results_as_list(it_iv_replacement_scores))
        logging.info('R-squared of class-pull plot: %f', r2)
        # Data currently rounded to 2 significant digits. Round to nearest int to make plot less cluttered
        myrange = plot_dots(*class_pull_results_as_list(round_class_pull_to_given_precision(it_iv_replacement_scores)))
        ax = plt.gca()
        # ax.set_ylim([-15, 15])
        # ax.set_xlim([-15, 15])
        plt.title('y=%.2fx%+.2f; r2=%.2f(%.2f); w=%s--%s' % (coef[0], coef[1], r2, r2adj, myrange[0], myrange[1]))

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
        else:
            pass
        # # use the space for something else
        # if params.class_pull:
        #     # remove features with a low class pull and repeat analysis
        #     x, y, z = class_pull_results_as_list(it_iv_replacement_scores)
        #     x1, y1, z1 = [], [], []
        #     for xv, yv, zv in zip(x, y, z):
        #         if not -4 < xv < 4:
        #             x1.append(xv)
        #             y1.append(yv)
        #             z1.append(zv)
        #
        #     if x1:  # filtering may remove all features
        #         plt.subplot(2, 3, 5)
        #         coef, r2, r2adj = plot_regression_line(x1, y1, z1)
        #         # Data currently rounded to 2 significant digits. Round to nearest int to make plot less cluttered
        #         myrange = plot_dots(x1, y1, z1)
        #         plt.title('y=%.2fx%+.2f; r2=%.2f(%.2f); w=%s--%s' % (coef[0], coef[1], r2,
        #                                                              r2adj, myrange[0], myrange[1]))

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
            if v == False:
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
        # do just one experiment, without any concurrency
        logging.info('Analysing just one experiment: %d', parameters.experiment)
        # do_work(parameters, parameters.experiment, 0, folds=20, workers=2, cursor=c)
        do_work(parameters, 0, 0, folds=2, workers=1, cursor=None)
        # do_work(parameters, 8, 0, folds=4, workers=1, cursor=None)

