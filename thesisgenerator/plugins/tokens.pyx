from functools import total_ordering
import logging
from operator import itemgetter


class DocumentFeature(object):
    def __init__(self, type, tokens):
        self.type = type
        self.tokens = tokens

    _TYPES = dict([ ('NVN', 'SVO'), ('JN', 'AN'), ('VN', 'VO'), ('NN', 'NN') ])

    @classmethod
    def from_string(cls, string):
        """
        Takes a string representing a DocumentFeature and creates and object out of it. String format is
        "word/PoS" or "word1/PoS1_word2/PoS2",... The type of the feature will be inferred from the length and
        PoS tags of the input string.

        From http://codereview.stackexchange.com/questions/38422/speeding-up-a-cython-program/38446?noredirect=1#38446

        :type string: str
        """
        try:
            tokens = string.strip().split('_')
            if len(tokens) > 3:
                raise ValueError('Document feature %s is too long' % string)

            tokens = [token.split('/') for token in tokens]

            # Check for too many slashes, too few slashes, or empty words
            if not all(map(lambda token: len(token) == 2 and token[0], tokens)):
                #raise ValueError('Invalid document feature %s' % string)
                return DocumentFeature('EMPTY', tuple())

            tokens = tuple(Token(word, pos) for (word, pos) in tokens)

            type = cls._TYPES.get(''.join([t.pos for t in tokens]),
                ('EMPTY', '1-GRAM', '2-GRAM', '3-GRAM')[len(tokens)])
        except:
            logging.error('Cannot create token out of string %s', string)
            raise

        return DocumentFeature(type, tokens)

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