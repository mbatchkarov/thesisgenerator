# coding=utf-8
from collections import defaultdict
import locale
import logging
from thesisgenerator.plugins.thesaurus_loader import get_thesaurus


def build_tokenizer(**kwargs):
    global tokenizer
    tokenizer = XmlTokenizer(**kwargs)
    return tokenizer


def get_tokenizer():
    return tokenizer


class XmlTokenizer(object):
    # for parsing integers with comma for thousands separator
    locale.setlocale(locale.LC_ALL, 'en_US')

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

    thes_entries = None

    def __init__(self, normalise_entities=False, use_pos=True,
                 coarse_pos=True, lemmatize=True,
                 lowercase=True, keep_only_IT=False):
        self.normalise_entities = normalise_entities
        self.use_pos = use_pos
        self.coarse_pos = coarse_pos
        self.lemmatize = lemmatize
        self.lowercase = lowercase
        self.keep_only_IT = keep_only_IT

    def __call__(self, doc):
        return self.tokenize(doc)

    def tokenize(self, doc):
        """
        Tokenizes a Stanford Core NLP processed document by parsing the XML and
        extracting tokens and their lemmas, with optional lowercasing
        If requested, the named entities will be replaced with the respective
         type, e.g. PERSON or ORG, otherwise numbers and punctuation will be
         canonicalised
        """

        if not self.thes_entries and self.keep_only_IT:
            thes_entries = set(get_thesaurus().keys())
            if not thes_entries:
                raise Exception('A thesaurus is required with keep_only_IT')
        try:
            import xml.etree.cElementTree as ET
        except ImportError:
            logging.getLogger().warn('cElementTree not available')
            import xml.etree.ElementTree as ET

        try:
            tree = ET.fromstring(doc.encode("utf8"))
            tokens = []
            for element in tree.findall('.//token'):
                if self.lemmatize:
                    txt = element.find('lemma').text
                else:
                    txt = element.find('word').text

                # check if the token is a number before things have been done
                #  to it
                am_i_a_number = self._is_number(txt)

                pos = element.find('POS').text.upper()
                if self.use_pos:
                    if self.coarse_pos:
                        pos = self.pos_coarsification_map[pos.upper()]
                    txt = '%s/%s' % (txt, pos)

                if self.normalise_entities:
                    try:
                        iob_tag = element.find('NER').text.upper()
                    except AttributeError:
                        logging.getLogger().error(
                            'You have requested named entity normalisation,'
                            ' but the input data is not annotated for '
                            'entities')
                        raise ValueError('Data not annotated for named '
                                         'entities')

                    if iob_tag != 'O':
                        txt = '__NER-%s__' % iob_tag

                if pos == 'PUNCT':
                    txt = '__PUNCT__'
                elif am_i_a_number:
                    txt = '__NUMBER__'

                if self.lowercase:
                    txt = txt.lower()

                if self.keep_only_IT and txt not in thes_entries:
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

        try:
            locale.atof(s)
            is_int = True
        except ValueError:
            is_int = False

        is_only_digits_or_punct = True
        for ch in s:
            if ch.isalpha():
                is_only_digits_or_punct = False
                break

        return is_float or is_int or is_only_digits_or_punct