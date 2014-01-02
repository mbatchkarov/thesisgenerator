# coding=utf-8
from StringIO import StringIO
from collections import defaultdict
from copy import deepcopy
from functools import total_ordering
import logging
from operator import itemgetter

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import networkx as nx

from thesisgenerator.classifiers import NoopTransformer


try:
    import xml.etree.cElementTree as ET
except ImportError:
    logging.warn('cElementTree not available')
    import xml.etree.ElementTree as ET


@total_ordering
class Token(object):
    def __init__(self, text, pos, index=0):
        self.text = text
        self.pos = pos
        self.index = index

    def __str__(self):
        return '{}/{}'.format(self.text, self.pos) if self.pos else self.text

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return (not self < other) and (not other < self)

    def __lt__(self, other):
        return (self.text, self.pos) < (other.text, other.pos)

    def __hash__(self):
        return hash((self.text, self.pos))


class DocumentFeature(object):
    def __init__(self, type, tokens):
        self.type = type
        self.tokens = tokens

    @classmethod
    def from_string(cls, string):
        """
        Takes a string representing a DocumentFeature and creates and object out of it. String format is
        "word/POS" or "word1/PoS1 word2/PoS2",... The type of the feature will be inferred from the length and
        PoS tags of the input string. Currently supports 1-GRAM, AN, NN, SVO and SV.

        :type string: str
        """
        try:
            token_count = string.count('_') + 1
            pos_count = string.count('/')
            if token_count != pos_count:
                return DocumentFeature('EMPTY', tuple())

            tokens = string.strip().split('_')
            if len(tokens) > 3:
                raise ValueError('Document feature %s is too long' % string)
            bits = [x.split('/') for x in tokens]
            if not all(map(itemgetter(0), bits)):
                # ignore tokens with no text
                return DocumentFeature('EMPTY', tuple())
            tokens = tuple(Token(word, pos) for (word, pos) in bits)

            if len(tokens) == 1:
                t = '1-GRAM'
            elif ''.join([t.pos for t in tokens]) == 'NVN':
                t = 'SVO'
            elif ''.join([t.pos for t in tokens]) == 'JN':
                t = 'AN'
            elif ''.join([t.pos for t in tokens]) == 'VN':
                t = 'VO'
            elif ''.join([t.pos for t in tokens]) == 'NN':
                t = 'NN'
            elif len(tokens) == 2:
                t = '2-GRAM'
            elif len(tokens) == 3:
                t = '3-GRAM'
            else:
                t = 'EMPTY'
        except:
            logging.error('Cannot create token out of string %s', string)
            raise

        return DocumentFeature(t, tokens)

    def tokens_as_str(self):
        """
        Represents the features of this document as a human-readable string
        DocumentFeature('1-GRAM', ('X', 'Y',)) -> 'X Y'
        """
        return '_'.join(str(t) for t in self.tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, item):
        '''
        Override slicing operator. Creates a feature from the tokens of this feature between
        positions beg (inclusive) and end (exclusive)
        :param beg:
        :type beg: int
        :param end:
        :type end: int or None
        :return:
        :rtype: DocumentFeature
        '''
        tokens = self.tokens[item]
        try:
            l = len(tokens)
            return DocumentFeature.from_string('_'.join(map(str, tokens)))
        except TypeError:
            # a single token has no len
            return DocumentFeature.from_string(str(tokens))


    def __str__(self):
        return '{}:{}'.format(self.type, self.tokens)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and self.__dict__ == other.__dict__)

    def __lt__(self, other):
        return (self.type, self.tokens) < (other.type, other.tokens)

    def __hash__(self):
        return hash((self.type, self.tokens))


class XmlTokenizer(object):
    # for parsing integers with comma for thousands separator
    # locale.setlocale(locale.LC_ALL, 'en_US')

    # copied from feature extraction toolkit
    pos_coarsification_map = defaultdict(lambda: "UNK")
    pos_coarsification_map.update({
        "JJ": "J",
        "JJN": "J",
        "JJS": "J",
        "JJR": "J",

        "VB": "V",
        "VBD": "V",
        "VBG": "V",
        "VBN": "V",
        "VBP": "V",
        "VBZ": "V",

        "NN": "N",
        "NNS": "N",
        "NNP": "N",
        "NPS": "N",
        "NP": "N",

        "RB": "RB",
        "RBR": "RB",
        "RBS": "RB",

        "DT": "DET",
        "WDT": "DET",

        "IN": "CONJ",
        "CC": "CONJ",

        "PRP": "PRON",
        "PRP$": "PRON",
        "WP": "PRON",
        "WP$": "PRON",

        ".": "PUNCT",
        ":": "PUNCT",
        ":": "PUNCT",
        "": "PUNCT",
        "'": "PUNCT",
        "\"": "PUNCT",
        "'": "PUNCT",
        "-LRB-": "PUNCT",
        "-RRB-": "PUNCT",

        # the four NE types that FET 0.3.6 may return as PoS tags
        # not really needed in the classification pipeline yet
        "PERSON": "PERSON",
        "LOC": "LOC",
        "ORG": "ORG",
        "NUMBER": "NUMBER"
    })

    def __init__(self, memory=NoopTransformer(), normalise_entities=False, use_pos=True,
                 coarse_pos=True, lemmatize=True, lowercase=True,
                 remove_stopwords=False, remove_short_words=False,
                 use_cache=False, dependency_format='collapsed-ccprocessed'):
        #store all important parameteres
        self.normalise_entities = normalise_entities
        self.use_pos = use_pos
        self.coarse_pos = coarse_pos
        self.lemmatize = lemmatize
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.remove_short_words = remove_short_words
        self.dependency_format = dependency_format

        # store the important parameters for use as joblib keys
        self.important_params = deepcopy(self.__dict__)

        self.charset = 'utf8'
        self.charset_error = 'replace'
        self.cached_tokenize_corpus = memory.cache(self._tokenize_corpus, ignore=['corpus', 'self'])
        self.cache_miss_count = 0

    def __setattr__(self, name, value):
        self.__dict__[name] = value

        # if we try to modify an essential field (one that affect the operation of the tokenizer) make sure to update
        # that in self.important_params. This is needed because once a tokenizer is created any modifications to its
        # important fields will not be propagated to self.important_params, practically undoing the modification if
        # caching is used
        if hasattr(self, 'important_params'):
            # before constructor is done self.important_params does not exist
            if name in self.important_params.keys():
                self.important_params[name] = value


    def _tokenize_corpus(self, corpus, corpus_id_joblib, **kwargs):
        # use the tokenizer settings (**kwargs) as cache key,
        # ignore the identity of the XmlTokenizer object

        # record how many times this tokenizer has had a cache miss
        self.cache_miss_count += 1
        return [self.tokenize_doc(x) for x in corpus]

    def tokenize_corpus(self, corpus, corpus_id_joblib):
        # externally visible method- uses a corpus identifier and a bunch of important tokenizer
        # settings to query the joblib cache
        return self.cached_tokenize_corpus(corpus, corpus_id_joblib, **self.important_params)

    def _process_sentence(self, tree):
        tokens = []
        for element in tree.findall('.//token'):
            if self.lemmatize:
                txt = element.find('lemma').text
            else:
                txt = element.find('word').text
                # check if the token is a number/stopword before things have been done to it
            am_i_a_number = self._is_number(txt)

            if self.remove_stopwords and txt.lower() in ENGLISH_STOP_WORDS:
                # logging.debug('Tokenizer ignoring stopword %s' % txt)
                continue

            if self.remove_short_words and len(txt) <= 3:
                # logging.debug('Tokenizer ignoring short word %s' % txt)
                continue

            pos = element.find('POS').text.upper() if self.use_pos else ''
            if self.coarse_pos:
                pos = self.pos_coarsification_map[pos.upper()]

            iob_tag = 'MISSING'
            if self.normalise_entities:
                try:
                    iob_tag = element.find('NER').text.upper()
                except AttributeError:
                    logging.error('You have requested named entity '
                                  'normalisation, but the input data are '
                                  'not annotated for entities')
                    raise ValueError('Data not annotated for named '
                                     'entities')

                if iob_tag != 'O':
                    txt = '__NER-%s__' % iob_tag
                    pos = '' # normalised named entities don't need a PoS tag

            if pos == 'PUNCT' or am_i_a_number:
                # logging.debug('Tokenizer ignoring stopword %s' % txt)
                continue

            if self.lowercase and '__NER' not in txt:
                txt = txt.lower()

            if '/' in txt or '_' in txt:
            # I use these chars as separators later, remove them now to avoid problems down the line
                if iob_tag in {'O', 'MISSING'}:
                    #logging.info('Funny token found: %s, pos is %s', txt, pos)
                    continue
                else:
                    # I put the underscore there, e.g. __NER-LOCATION__!
                    pass
            tokens.append(Token(txt, pos, int(element.get('id'))))

        # build a graph from the dependency information available in the input
        tokens_ids = set(x.index for x in tokens)
        dep_tree = nx.DiGraph()
        dep_tree.add_nodes_from(tokens_ids)

        dependencies = tree.find('.//{}-dependencies'.format(self.dependency_format))
        if dependencies:
            for dep in dependencies.findall('.//dep'):
                type = dep.get('type')
                head = dep.find('governor')
                head_idx = int(head.get('idx'))
                #head_txt = head.text

                dependent = dep.find('dependent')
                dependent_idx = int(dependent.get('idx'))
                #dependent_txt = dependent.text
                if dependent_idx in tokens_ids and head_idx in tokens_ids:
                    dep_tree.add_edge(head_idx, dependent_idx, type=type)
                    #a=nx.draw(dep_tree, nx.graphviz_layout(dep_tree,prog='dot'), font_size=8, node_size=500,
                    #          edge_labels = [x[2]['type'] for x in dep_tree.edges(data=True)])
                    #import matplotlib.pyplot as plt
                    #plt.savefig("atlas.png",dpi=275)
                    #else:
                    #t = ET.ElementTree(tree)
                    #s = StringIO()
                    #t.write(s)
                    #logging.info('Cant find dependency info in sentence: \n %s', s.getvalue())

        return tokens, (dep_tree, {t.index: t for t in tokens})

    def tokenize_doc(self, doc, **kwargs):
        """
        Tokenizes a Stanford Core NLP processed document by parsing the XML and
        extracting tokens and their lemmas, with optional lowercasing
        If requested, the named entities will be replaced with the respective
         type, e.g. PERSON or ORG, otherwise numbers and punctuation will be
         canonicalised

         :returns: a list of sentence tuple of the form (tokens_list,
                                                (dependency_graph, {token index in sentence -> token object})
                                                )
        """


        # decode document
        doc = doc.decode(self.charset, self.charset_error)
        try:
            tree = ET.fromstring(doc.encode("utf8"))
            sentences = []
            for sent_element in tree.findall('.//sentence'):
                sentences.append(self._process_sentence(sent_element))
        except ET.ParseError as e:
            logging.error('Parse error %s', e)
            pass
            # on OSX the .DS_Store file is passed in, if it exists
            # just ignore it
        return sentences

    def __str__(self):
        return 'XmlTokenizer:{}'.format(self.important_params)

    @staticmethod
    def _is_number(s):
        """
        Checks if the given string is an int or a float. Numbers with thousands
        separators (e.g. "1,000.12") are also recognised. Returns true of the string
        contains only digits and punctuation, e.g. 12/23
        """
        try:
            float(s)
            is_float = True
        except ValueError:
            is_float = False

        # try:
        #     locale.atof(s)
        #     is_int = True
        # except ValueError:
        #     is_int = False

        is_only_digits_or_punct = True
        for ch in s:
            if ch.isalpha():
                is_only_digits_or_punct = False
                break

        return is_float or is_only_digits_or_punct #or is_int