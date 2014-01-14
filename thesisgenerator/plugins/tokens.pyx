from functools import total_ordering
import logging
from operator import itemgetter
from itertools import izip_longest
import re


class DocumentFeature(object):
    def __init__(self, type, tokens):
        self.type = type
        self.tokens = tokens

    _TYPES = dict([('NVN', 'SVO'), ('JN', 'AN'), ('VN', 'VO'), ('NN', 'NN')])
    #  not an underscore + text + underscore or end of line
    #  see re.split documentation on capturing (the first two) and non-capturing groups (the last one)
    _TOKEN_RE = re.compile(r'([^/_]+)/([A-Z]+)(?:_|$)')

    @classmethod
    def from_string(cls, string):
        try:
            match = cls._TOKEN_RE.split(string, 3)
            type = ''.join(match[2::3])
            match = iter(match)
            tokens = []
            for (junk, word, pos) in izip_longest(match, match, match):
                if junk:        # Either too many tokens, or invalid token
                    raise ValueError(junk)
                if not word:
                    break
                tokens.append(Token(word, pos))
            type = cls._TYPES.get(type,
                                  ('EMPTY', '1-GRAM', '2-GRAM', '3-GRAM')[len(tokens)])
            return DocumentFeature(type, tuple(tokens))
        except:
            logging.error('Cannot create token out of string %s', string)
            return DocumentFeature('EMPTY', tuple())

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


@total_ordering
class Token(object):
    def __init__(self, text, pos, index=0, ner=None):
        self.text = text
        self.pos = pos
        self.index = index
        self.ner = ner

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