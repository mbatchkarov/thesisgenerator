# coding=utf-8
from collections import defaultdict
# import locale
from copy import deepcopy
import logging
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

try:
    import xml.etree.cElementTree as ET
except ImportError:
    logging.warn('cElementTree not available')
    import xml.etree.ElementTree as ET


def build_tokenizer(**kwargs):
    global tokenizer
    tokenizer = XmlTokenizer(**kwargs)
    return tokenizer


def get_tokenizer():
    return tokenizer


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
    })

    def __init__(self, memory, normalise_entities=False, use_pos=True,
                 coarse_pos=True, lemmatize=True,
                 lowercase=True, keep_only_IT=False, thesaurus=defaultdict(list),
                 remove_stopwords=False, remove_short_words=False,
                 use_cache=False):
        self.normalise_entities = normalise_entities
        self.use_pos = use_pos
        self.coarse_pos = coarse_pos
        self.lemmatize = lemmatize
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.remove_short_words = remove_short_words
        try:
            self.thes_names = tuple(thesaurus.thesaurus_names)
        except AttributeError:
            self.thes_names = ''
        self.keep_only_IT = keep_only_IT

        # guard against using an empty thesaurus
        if not thesaurus and keep_only_IT:
            raise Exception('A thesaurus is required with keep_only_IT')

        self.param_values = deepcopy(self.__dict__)
        if not self.keep_only_IT:
            # thesaurus names are unimportant if tokenizer isn't using them
            del self.param_values['thes_names']

        self.thes_entries = set(thesaurus.keys())
        self.cached_tokenize = memory.cache(self.noncached_tokenize, ignore=['self'])

    def tokenize(self, doc):
        # also use the tokenizer settings as cache key,
        # ignore the identity of the XmlTokenizer object
        return self.cached_tokenize(doc, **self.param_values)

    def noncached_tokenize(self, doc, **kwargs):
        """
        Tokenizes a Stanford Core NLP processed document by parsing the XML and
        extracting tokens and their lemmas, with optional lowercasing
        If requested, the named entities will be replaced with the respective
         type, e.g. PERSON or ORG, otherwise numbers and punctuation will be
         canonicalised
        """

        print 'called with **', kwargs

        try:
            tree = ET.fromstring(doc.encode("utf8"))
            tokens = []
            for element in tree.findall('.//token'):
                if self.lemmatize:
                    txt = element.find('lemma').text
                else:
                    txt = element.find('word').text

                # check if the token is a number/stopword before things have
                # been done to it
                am_i_a_number = self._is_number(txt)

                if self.remove_stopwords and txt.lower() in ENGLISH_STOP_WORDS:
                    # logging.debug('Tokenizer ignoring stopword %s' % txt)
                    continue

                if self.remove_short_words and len(txt) <= 3:
                    # logging.debug('Tokenizer ignoring short word %s' % txt)
                    continue

                pos = element.find('POS').text.upper()
                if self.use_pos:
                    if self.coarse_pos:
                        pos = self.pos_coarsification_map[pos.upper()]
                    txt = '%s/%s' % (txt, pos)

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

                if pos == 'PUNCT' or am_i_a_number:
                    # logging.debug('Tokenizer ignoring stopword %s' % txt)
                    continue

                if self.lowercase:
                    txt = txt.lower()

                if self.keep_only_IT and txt not in self.thes_entries:
                    # logging.debug('Tokenizer ignoring OOT token: %s' % txt)
                    continue
                tokens.append(txt)

        except ET.ParseError:
            pass
            # on OSX the .DS_Store file is passed in, if it exists
            # just ignore it
        return tokens


    def _is_number(self, s):
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