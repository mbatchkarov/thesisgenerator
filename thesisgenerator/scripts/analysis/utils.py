#####################################################################
# UTILITY FUNCTIONS
#####################################################################
from itertools import groupby
import logging
from operator import itemgetter
import pickle


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
        stats = pickle.load(infile)
    try:
        flp = stats.nb_feature_log_prob
        inv_voc = stats.nb_inv_voc
        return flp, inv_voc
    except AttributeError:
        logging.warn('Classifier parameters unavailable')
        return None, None


def get_experiment_info_string(cursor, exp_num, subexp_name):
    query = 'SELECT score_mean, score_std FROM data%d WHERE NAME="%s" AND ' \
            'metric="macroavg_f1" AND classifier="MultinomialNB"' % (exp_num, subexp_name)
    cursor.execute(query)
    f1 = 'F1=%.2f+-%1.2f' % cursor.fetchall()[0]
    query = "SELECT * FROM ExperimentDescriptions WHERE number=%d" % exp_num
    cursor.execute(query)
    settings_data = cursor.fetchone()
    desc = cursor.description
    settings = {}
    for (key, value) in zip(desc, settings_data):
        settings[key[0]] = value
    settings_string = '\n'.join('%s:%s' % (str(k), str(v)) for k, v in settings.items())
    return '%s:\n %s\n%s' % (subexp_name, settings_string, f1)