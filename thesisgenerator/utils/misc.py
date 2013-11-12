import inspect
from itertools import tee, izip
from configobj import ConfigObj
from thesisgenerator.utils.reflection_utils import get_named_object

__author__ = 'mmb28'


def _walk_pairwise(iterable):
    """
    s -> (s0,s1), (s1,s2), (s2, s3), ...

    From http://docs.python.org/2/library/itertools.html
    """

    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)


def get_susx_mysql_conn():
    import MySQLdb as mdb

    config = ConfigObj('thesisgenerator/db-credentials')
    if not config:
        return None
        # thesisgenerator/db-credentials file not found. This is needed for a
        # MySQL connection

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


def noop(x):
    return x


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