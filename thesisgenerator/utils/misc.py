import inspect
import os
import sqlite3
import errno
from thesisgenerator.utils.reflection_utils import get_named_object
import logging
import numpy as np


def get_susx_mysql_conn():
    """
    Returns a mysql connection to the Sussex BoV database, or None if
     - MySQLdb is not installed
     - the db-credentials is not present
    :return:
    """
    return sqlite3.connect('./db.sqlite')


def linear_compress(x):
    if x <= 0.1:
        return 0.
    elif x >= 0.3:
        return 1.
    else:
        return 5 * x - 0.5  # f(0.1) = 0 and f(0.3) = 1


def unit(x):
    return x


def noop(*args, **kwargs):
    pass


def one(*args, **kwargs):
    return 1.


class ChainCallable(object):
    """
    Chains several functions as specified in the configuration. When called
    returns the return values of all of its callables as a tuple

    Example usage:

    # some sample functions
    def t1(shared_param, conf):
        print 'called t1(%s, %s)' % (str(shared_param), str(shared_param))
        return 1


    def t2(shared_param, conf):
        print 'called t1(%s, %s)' % (str(shared_param), str(shared_param))
        return 2

    config = {'a': 1, 'b': 2}
    ccall = chain_callable(config)
    print ccall('X_Y')
    print ccall('A_B')

    """

    def __init__(self, config):
        self.config = config

    def __call__(self, true_labels, predicted_labels):
        to_call = [(x, get_named_object(x)) for x in
                   self.config.keys() if self.config[x]['run']]
        options = {}
        result = {}
        for func_name, func in to_call:
            initialize_args = inspect.getargspec(func)[0]
            call_args = {arg: val for arg, val in self.config[func_name].items()
                         if val != '' and arg in initialize_args}
            options[func_name] = call_args
            result[func_name.strip()] = (
                func(true_labels, predicted_labels, **call_args))
        return result


def calculate_log_odds(X, y, column_indices=None):
    """

    :param X: term-document matrix, shape (m,n)
    :type X: scipy.sparse
    :param y: document labels, shape (m,)
    :type y: np.array
    :param column_indices: compute score only for these columns, other will be set to 0
    :return: log odds scores of all features
    :rtype: array-like, shape (n,)
    """
    alpha = .000001
    alpha_denom = alpha * len(set(y))

    log_odds = np.empty(X.shape[1])
    class0_indices = y == sorted(set(y))[0]
    if column_indices is None:
        column_indices = np.arange(X.shape[1])
    for idx in column_indices:
        all_counts = X[:, idx]  # document counts of this feature
        total_counts = np.count_nonzero(all_counts.data)  # how many docs the feature occurs in
        count_in_class0 = np.count_nonzero(all_counts[class0_indices].data)  # how many of them are class 0

        # p = float(count_in_class0) / total_counts
        # smoothing to avoid taking log(0) below
        p = (float(count_in_class0) + alpha) / (total_counts + alpha_denom)
        log_odds_this_feature = np.log(p) - np.log(1 - p)
        log_odds[idx] = log_odds_this_feature
    return log_odds


def update_dict_according_to_mask(v, mask):
    """
    Given a dictionary of {something:index} and a boolean mask, removes items as specified by the mask
    and re-assigns consecutive indices to the remaining items. The values of `v` are assumed to be
    consecutive integers starting at 0
    :param mask: array-like, must be indexable by the values of `v`
    :rtype: dict
    """
    if len(v) != len(mask):
        logging.error('Mask and dict do not match in size: %d vs %d', mask.shape[0], len(v))
        return

    # see which features are left
    v = {feature: index for feature, index in v.items() if mask[index]}
    # assign new indices for each remaining feature in order, map: old_index -> new_index
    new_indices = {old_index: new_index for new_index, old_index in enumerate(sorted(v.values()))}
    # update indices in vocabulary
    return {feature: new_indices[index] for feature, index in v.items()}


def force_symlink(existing_file, link_name):
    """
    Created a symlink. If `link_name` exists it is deleted first. Take care!
    :raise e:
    """
    try:
        os.symlink(existing_file, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(existing_file, link_name)
        else:
            raise e