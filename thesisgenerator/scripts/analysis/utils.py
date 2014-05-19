from collections import Counter
from itertools import groupby
import logging
from operator import itemgetter
import pickle
import numpy as np
import scipy.sparse as sp

#####################################################################
# UTILITY FUNCTIONS
#####################################################################


def class_pull_results_as_list(replacement_scores):
    """
    Expands a dict into three lists. Dict must have the following structure
     (key1: float, key2: float): value:float
    >>> class_pull_results_as_list({(1.1, 2.1):3, (3.1, 4.1):4})
    ([1.1, 3.1], [2.1, 4.1], [3, 4])
    """
    x = []
    y = []
    z = []
    for (orig_value, repl_value), repl_count in replacement_scores.iteritems():
        y.append(repl_value)
        x.append(orig_value)
        z.append(repl_count)
    return x, y, z


def round_class_pull_to_given_precision(scores, xprecision=0, yprecision=0):
    """
    Rounds keys in dict to nearest integer. Dict must have the following structure
     (key1: float, key2: float): value:float

    Entries that fall into the same bin after rounding are added up, e.g.
    >>> round_class_pull_to_given_precision({(1.1, 2.1):3, (1.111, 2.111):3})
    {(1.0, 2.0): 6}
    >>> round_class_pull_to_given_precision({(1.1, 2.1):3, (1.111, 2.111):3}, 1, 1)
    {(1.1, 2.1): 6}
    """
    s = [(round(a, xprecision), round(b, yprecision), c) for ((a, b), c) in scores.items()]
    s = sorted(s, key=itemgetter(0, 1))
    rounded_scores = {}
    for key, group in groupby(s, itemgetter(0, 1)):
        rounded_scores[key] = sum(x[2] for x in group)
    return rounded_scores


def load_classificational_vectors(pickle_file):
    with open(pickle_file) as infile:
        b = pickle.load(infile)

    feature_counts_in_tr_set = np.array(b.tr_matrix.sum(axis=0)).ravel()

    voc_size = len(b.inv_voc)
    mat = sp.lil_matrix((voc_size, voc_size))
    mat.setdiag(np.ones((voc_size,)))
    probabilities = b.clf.predict_log_proba(mat.tocsr())

    return probabilities, b.inv_voc, feature_counts_in_tr_set


def get_good_vectors(all_clf_vectors, feature_counts_in_tr_set, min_freq, inv_voc, thes):
    exceeds_threshold = feature_counts_in_tr_set >= min_freq
    not_unigram = [inv_voc[idx].type != '1-GRAM' for idx in range(len(inv_voc))]
    in_thes = [inv_voc[idx].tokens_as_str() in thes for idx in range(len(inv_voc))]
    mask_to_keep = np.logical_and(np.logical_and(in_thes, not_unigram),
                                  exceeds_threshold)
    logging.info('%d total features. Types are %r. %d are IT.',
                 len(inv_voc),
                 Counter(x.type for x in inv_voc.values()),
                 sum(in_thes))
    logging.info('The types of those IT are %r', Counter(x.type for x in inv_voc.values() if x.tokens_as_str() in thes))
    logging.info('%d/%d features are considered good (IV, IT, NP and frequent)', sum(mask_to_keep), len(inv_voc))

    # keys need to be consecutive, but the selection above will remove some of them. Assign new consecutive keys
    # to the remaining values (in order), eg. {1:a, 2:b, 3:c} becomes {1:a, 2:c}
    pruned_inv_voc = {idx: inv_voc[idx] for idx, keep_this in enumerate(mask_to_keep) if keep_this}
    new_inv_voc = {new_index: pruned_inv_voc[old_index] for new_index, old_index in
                   enumerate(sorted(pruned_inv_voc.keys()))}

    logging.info('Their types are %r', Counter(x.type for x in new_inv_voc.values()))

    return all_clf_vectors[mask_to_keep, :], new_inv_voc  # feature counts


def get_experiment_info_string(cursor, exp_num, subexp_name):
    query = 'SELECT score_mean, score_std FROM data%d WHERE NAME="%s" AND ' \
            'metric="macroavg_f1" AND classifier="MultinomialNB"' % (exp_num, subexp_name)
    cursor.execute(query)
    f1 = 'F1=%.2f+-%1.2f' % cursor.fetchall()[0]
    query = "SELECT * FROM ExperimentDescriptions WHERE id=%d" % exp_num
    cursor.execute(query)
    settings_data = cursor.fetchone()
    desc = cursor.description
    if not settings_data:
        logging.warn('Description of Experiment %d not found in database, using dummy values')
        settings_data = '-' * len(desc)
    settings = {}
    for (key, value) in zip(desc, settings_data):
        settings[key[0]] = value
    settings_string = '\n'.join('%s:%s' % (str(k), str(v)) for k, v in settings.items())
    return '%s:\n %s\n%s' % (subexp_name, settings_string, f1)