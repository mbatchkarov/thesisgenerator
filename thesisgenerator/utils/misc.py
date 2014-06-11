import inspect
from configobj import ConfigObj
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
    try:
        import MySQLdb as mdb
    except ImportError:
        logging.warn('MySQLdb not installed')
        return None

    config = ConfigObj('thesisgenerator/db-credentials')
    if not config:
        logging.warn('File thesisgenerator/db-credentials file not found. This is '
                     'needed for a MySQL connection. Check your working directory.')
        return None

    return mdb.connect(config['server'],
                       config['user'],
                       config['pass'],
                       config['db'])


def linear_compress(x):
    if x <= 0.1:
        return 0.
    elif x >= 0.3:
        return 1.
    else:
        return 5 * x - 0.5 # f(0.1) = 0 and f(0.3) = 1


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
        self.to_call = [(x, get_named_object(x)) for x in
                        config.keys() if config[x]['run']]
        self.config = config

    def __call__(self, true_labels, predicted_labels):
        options = {}
        result = {}
        for func_name, func in self.to_call:
            initialize_args = inspect.getargspec(func)[0]
            call_args = {arg: val for arg, val in self.config[func_name].items()
                         if val != '' and arg in initialize_args}
            options[func_name] = call_args
            result[func_name.strip()] = (
                func(true_labels, predicted_labels, **call_args))
        return result

def calculate_log_odds(X, y):
    """

    :param X: term-document matrix, shape (m,n)
    :type X: array-like
    :param y: document labels, shape (m,)
    :type y: array-like
    :return: log odds scores of all features
    :rtype: array-like, shape (n,)
    """
    log_odds = np.empty(X.shape[1])
    class0_indices = y == sorted(set(y))[0]
    X = X.A
    for idx in range(X.shape[1]):
        all_counts = X[:, idx]  # document counts of this feature
        total_counts = np.count_nonzero(all_counts)  # how many docs the feature occurs in
        count_in_class0 = np.count_nonzero(all_counts[class0_indices])  # how many of them are class 0
        p = float(count_in_class0) / total_counts
        log_odds_this_feature = np.log(p) - np.log(1 - p)
        log_odds[idx] = log_odds_this_feature
    return log_odds