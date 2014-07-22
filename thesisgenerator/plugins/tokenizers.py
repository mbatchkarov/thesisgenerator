# coding=utf-8
from collections import defaultdict
from copy import deepcopy
import logging

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import networkx as nx

from thesisgenerator.classifiers import NoopTransformer
from discoutils.tokens import Token


try:
    import xml.etree.cElementTree as ET
except ImportError:
    logging.warning('cElementTree not available')
    import xml.etree.ElementTree as ET


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
                 remove_long_words=False,
                 use_cache=False, dependency_format='collapsed-ccprocessed'):
        #store all important parameteres
        self.normalise_entities = normalise_entities
        self.use_pos = use_pos
        self.coarse_pos = coarse_pos
        self.lemmatize = lemmatize
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.remove_short_words = remove_short_words
        self.remove_long_words = remove_long_words
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
                continue

            if self.remove_long_words and len(txt) >= 25:
                continue

            pos = element.find('POS').text.upper() if self.use_pos else ''
            if self.coarse_pos:
                pos = self.pos_coarsification_map[pos.upper()]

            if pos == 'PUNCT' or am_i_a_number:
                # logging.debug('Tokenizer ignoring stopword %s' % txt)
                continue

            try:
                iob_tag = element.find('NER').text.upper()
            except AttributeError:
                # logging.error('You have requested named entity '
                #               'normalisation, but the input data are '
                #               'not annotated for entities')
                iob_tag = 'MISSING'
                # raise ValueError('Data not annotated for named entities')

            if '/' in txt or '_' in txt:
                # I use these chars as separators later, remove them now to avoid problems down the line
                logging.debug('Funny token found: %s, pos is %s', txt, pos)
                continue

            if self.lowercase:
                txt = txt.lower()

            if self.normalise_entities:
                if iob_tag != 'O':
                    txt = '__NER-%s__' % iob_tag
                    pos = ''  # normalised named entities don't need a PoS tag

            tokens.append(Token(txt, pos, int(element.get('id')), ner=iob_tag))

        # build a graph from the dependency information available in the input
        tokens_ids = set(x.index for x in tokens)
        dep_tree = nx.DiGraph()
        dep_tree.add_nodes_from(tokens_ids)

        dependencies = tree.find('.//{}-dependencies'.format(self.dependency_format))
        # some file are formatted like so: <basic-dependencies> ... </basic-dependencies>
        if not dependencies:
            # and some like so: <dependencies type="basic-dependencies"> ... </dependencies>
            # if one fails try the other. If that fails too something is wrong- perhaps corpus has not been parsed?
            dependencies = tree.find(".//dependencies[@type='{}-dependencies']".format(self.dependency_format))
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
        try:
            # tree = ET.fromstring(doc.encode("utf8"))
            tree = ET.fromstring(doc)
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

        return is_float or is_only_digits_or_punct  #or is_int