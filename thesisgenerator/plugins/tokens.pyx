from functools import total_ordering
import logging
from operator import itemgetter


class DocumentFeature(object):
    def __init__(self, type, tokens):
        self.type = type
        self.tokens = tokens

    @classmethod
    def from_string(cls, char* string):
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