# coding=utf-8

"""
A collection of random useful utilities
"""
from itertools import combinations
import re, numpy as np
import gzip

try:
    from xml.etree import cElementTree as ET
except ImportError:
    from xml.etree import ElementTree as ET


def get_named_object(pathspec):
    """Return a named from a module.
    """
    parts = pathspec.split('.')
    module = ".".join(parts[:-1])
    mod = __import__(module, fromlist = parts[-1])
    named_obj = getattr(mod, parts[-1])
    return named_obj


class GorkanaXmlParser():
    def __init__(self, source):
        self._source = source

    def documents(self):
        with gzip.open(self._source, 'r') as _in_fh:
            self._xml_etree = ET.iterparse(_in_fh, events = ('end',))
            regex = re.compile(
                '(?:&lt;|<)headline(?:&gt;|>)(.*)(?:&lt;|<)/headline(?:&gt;|>)')
            for _, element in self._xml_etree:
                if element.tag == 'documents' or element.text == None: continue

                article_text = element.text
                _headline = regex.findall(article_text)
                _headline = _headline[0] if len(_headline) > 0 else ''
                _body = regex.sub('', article_text)

                yield '%s\n%s' % (_headline.strip(), _body.strip())

    def targets(self):
        with gzip.open(self._source, 'r') as _in_fh:
            self._xml_etree = ET.iterparse(_in_fh, events = ('end',))
            for _, element in self._xml_etree:
                if element.tag == 'documents': continue
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
            test_index = np.zeros(n, dtype = np.bool)
            test_index[np.array(idx)] = True
            #            train_index = np.logical_not(test_index)
            train_index = test_index
            if self.indices:
                ind = np.arange(n)
                train_index = ind[train_index]
                test_index = ind[test_index]
            yield train_index, test_index