# coding=utf-8

"""
A collection of random useful utilities
"""
import fileinput
import inspect
from itertools import combinations
import re
import numpy as np
import gzip
import sys

try:
    from xml.etree import cElementTree as ET
except ImportError:
    from xml.etree import ElementTree as ET


def get_named_object(pathspec):
    """Return a named from a module.
    """
    parts = pathspec.split('.')
    module = ".".join(parts[:-1])
    mod = __import__(module, fromlist=parts[-1])
    named_obj = getattr(mod, parts[-1])
    return named_obj


def replace_in_file(file, search_exp, replace_exp):
   for line in fileinput.input(file, inplace=1):
       line = re.sub(search_exp, replace_exp, line)
       sys.stdout.write(line)


class GorkanaXmlParser(object):
    def __init__(self, source):
        self._source = source

    def documents(self):
        with gzip.open(self._source, 'r') as _in_fh:
            self._xml_etree = ET.iterparse(_in_fh, events=('end',))
            regex = re.compile(
                '(?:&lt;|<)headline(?:&gt;|>)(.*)(?:&lt;|<)/headline(?:&gt;|>)')
            for _, element in self._xml_etree:
                if element.tag == 'documents' or element.text is None: continue

                article_text = element.text
                _headline = regex.findall(article_text)
                _headline = _headline[0] if len(_headline) > 0 else ''
                _body = regex.sub('', article_text)

                yield '%s\n%s' % (_headline.strip(), _body.strip())

    def targets(self):
        with gzip.open(self._source, 'r') as _in_fh:
            self._xml_etree = ET.iterparse(_in_fh, events=('end',))
            for _, element in self._xml_etree:
                if element.tag == 'documents' or element.text is None: continue
                target = element.attrib['relevant'] == 'True'
                yield target


def gorkana_200_seen_positives_validation(x, y):
    i = 0
    pos = 0
    while pos < 200:
        i += 1
        pos += 1 if y[i] == 1 else 0

    return [(i, len(y))]


class LeaveNothingOut(object):
    """A modified version of sklearn.cross_validation.LeavePOut which leaves
    nothing out, i.e. the whole dataset it used for both testing and training
    """

    def __init__(self, n, indices=True):
        self.n = n
        self.indices = indices

    def __iter__(self):
        n = self.n
        comb = combinations(range(n), n)
        for idx in comb:
            test_index = np.zeros(n, dtype=np.bool)
            test_index[np.array(idx)] = True
            #            train_index = np.logical_not(test_index)
            train_index = test_index
            if self.indices:
                ind = np.arange(n)
                train_index = ind[train_index]
                test_index = ind[test_index]
            yield train_index, test_index


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
            call_args = {arg: val for arg, val in self.config[func_name]
            .items() if  val != '' and arg in initialize_args}
            options[func_name] = call_args
            result[func_name.strip()] = (
                func(true_labels, predicted_labels, **call_args))
        return result