from collections import defaultdict
from operator import itemgetter
import os
import pickle
import scipy.sparse as sp
from sklearn.feature_extraction.text import  TfidfVectorizer
from thesis_generator import config
from thesis_generator.__main__ import _config_logger
import xml.etree.ElementTree as ET

def _configure_logger():
    args = config.arg_parser.parse_args()
    log_path = os.path.join(args.log_path, 'bov-vectorizer')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    return _config_logger(log_path)


class ThesaurusVectorizer(TfidfVectorizer):
    """
    A thesaurus-backed CountVectorizer that replaces unknown features with
    their k nearest neighbours in the thesaurus
    """

    def __init__(self, thesaurus_file=None, k=1, sim_threshold=0.2,
                 lemmatize=False, log_vocabulary=False, coarse_pos=True,
                 input='content', charset='utf-8', charset_error='strict',
                 strip_accents=None, lowercase=True, use_pos=False,
                 preprocessor=None, tokenizer=None, analyzer='better',
                 stop_words=None, token_pattern=ur"(?u)\b\w\w+\b", min_n=None,
                 max_n=None, ngram_range=(1, 1), max_df=1.0, min_df=2,
                 max_features=None, vocabulary=None, binary=False, dtype=long,
                 norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False):
        """
        Builds a vectorizer the way a TfidfVectorizer is built, and takes one
        extra param specifying the path the the Byblo-generated thesaurus
        """
        try:
            self.thesaurus_file = thesaurus_file
            self.k = k
            self.sim_threshold = sim_threshold
            self.log_vocabulary = log_vocabulary # if I should log the
            self.lemmatize = lemmatize
            self.use_pos = use_pos
            self.coarse_pos = coarse_pos
            # vocabulary
            self.log_vocabulary_already = False #have I done it already
        except KeyError:
            pass
        super(ThesaurusVectorizer, self).__init__(input=input, charset=charset,
                                                  charset_error=charset_error,
                                                  strip_accents=strip_accents,
                                                  lowercase=lowercase,
                                                  preprocessor=preprocessor,
                                                  tokenizer=tokenizer,
                                                  analyzer=analyzer,
                                                  stop_words=stop_words,
                                                  token_pattern=token_pattern,
                                                  min_n=min_n,
                                                  max_n=max_n,
                                                  ngram_range=ngram_range,
                                                  max_df=max_df, min_df=min_df,
                                                  max_features=max_features,
                                                  vocabulary=vocabulary,
                                                  binary=False,
                                                  dtype=long, norm='l2',
                                                  use_idf=True,
                                                  smooth_idf=True,
                                                  sublinear_tf=False
        )

    def load_thesaurus(self):
        """
        Loads a Byblo-generated thesaurus form the specified file. If the file
        has been parsed already returns a cached version.

        Parameters:
        path: string, path the the Byblo-generated thesaurus
        pos_insensitive: boolean, whether the PoS tags should be stripped from
            entities (if they are present)
        sim_threshold: what is the min similarity for neighbours that should be
        loaded
        """
        path = self.thesaurus_file
        if not path:
            return None
        if path in thesauri:
            logger.info('Returning cached thesaurus for %s' % path)
            return thesauri[path]
        else:
            logger.info('Loading thesaurus %s from disk' % path)
            logger.info(
                'PoS: %r, coarse: %r, threshold %r' % (self.use_pos,
                                                       self.coarse_pos,
                                                       self.sim_threshold))
            FILTERED = '___FILTERED___'.lower()
            neighbours = defaultdict(list)
            with open(path) as infile:
                for line in infile:
                    tokens = line.strip().split('\t')
                    if len(tokens) % 2 == 0:
                    #must have an odd number of things, one for the entry and
                    # pairs for (neighbour, similarity)
                        continue
                    if tokens[0] != FILTERED:
                        indices = range(1, len(tokens), 2)
                        indices.insert(0,0)
                        for i in indices:# go over words
                            s = tokens[i].split('/')
                            if not self.use_pos:
                                s = s[:-1]    # remove PoS tag from token
                            if self.lowercase:
                                s[0] = s[0].lower()
                            tokens[i] = '/'.join(s)

                        to_insert = [(word, float(sim)) for (word, sim)
                                     in
                                     self.iterate_nonoverlapping_pairs(
                                         tokens, 1)
                                     if
                                     word != FILTERED and sim >
                                     self.sim_threshold]
                        # the step above may filter out all neighbours of an
                        # entry. if this happens, do not bother adding it
                        if len(to_insert) > 0:
                            if tokens[0] in neighbours:
                                logger.debug(
                                    'Multiple entries for "%s" found' %
                                    tokens[0])
                            neighbours[tokens[0]].extend(to_insert)

            for k, v in neighbours.iteritems():
                neighbours[k] = sorted(v, key=itemgetter(1))
                # todo this does not remove duplicate neighbours,
                # e.g. in thesaurus 1-1 "Jihad" has neighbours	Hamas and HAMAS,
                # which get conflated. Also, entries HEBRON and Hebron exist,
                # which need to be merged properly. Such events are quite
                # infrequent- 167/5700 = 3% entries in exp1-1 collide

            thesauri[path] = neighbours
            logger.info('Thesaurus contains %d entries' % len(neighbours))
            return neighbours


    def iterate_nonoverlapping_pairs(self, iterable, beg):
        for i in xrange(beg, len(iterable) - 1, 2):  #step size 2
            yield (iterable[i], iterable[i + 1])


    def my_feature_extractor(self, tokens, stop_words=None,
                             ngram_range=(1, 1), lemmatize=False):
        """
        Turn a document( a list of tokens) into a sequence of features. These
        include n-grams after stop words filtering,
        suffix features and shape features
        Based on sklearn.feature_extraction.text._word_ngrams
        """

        #todo add/enable feature functions here
        # handle stop words and lowercasing- this is needed because thesaurus
        # only contains lowercase entries
        if stop_words is not None:
            tokens = [w.lower() for w in tokens if
                      w not in stop_words and len(w) >
                      3]

        #    last_chars = ['**suffix(%s)' % token[-1] for token in tokens]
        #    shapes = ['**shape(%s)' % "".join(
        #        'x' if l.islower() else '#' if l.isdigit()  else 'X' for l in
        # token)
        #              for token in
        #              tokens]

        # handle token n-grams
        min_n, max_n = ngram_range
        if max_n != 1:
            original_tokens = tokens
            tokens = []
            n_original_tokens = len(original_tokens)
            for n in xrange(min_n,
                            min(max_n + 1, n_original_tokens + 1)):
                for i in xrange(n_original_tokens - n + 1):
                    tokens.append(u" ".join(original_tokens[i: i + n]))

        return tokens  # + last_chars + shapes


    def my_analyzer(self):
        return lambda doc: self.my_feature_extractor(
            doc, None, None)

    def build_analyzer(self):
        """
        Return a callable that handles preprocessing,
        tokenization and any additional feature extraction. Extends
        sklearn.feature_extraction.text.CountVectorizer.build_analyzer() by
        adding a 'better' option, which
        invokes self.my_feature_extractor, a more general function than
        CountVectorizer._word_ngrams()
        """
        if hasattr(self.analyzer, '__call__'):
            return self.analyzer

        preprocess = self.build_preprocessor()

        if self.analyzer == 'char':
            return lambda doc: self._char_ngrams(preprocess(self.decode(doc)))

        elif self.analyzer == 'char_wb':
            return lambda doc: self._char_wb_ngrams(
                preprocess(self.decode(doc)))

        elif self.analyzer == 'word':
            stop_words = self.get_stop_words()
            tokenize = self.build_tokenizer()

            return lambda doc: self._word_ngrams(
                tokenize(preprocess(self.decode(doc))), stop_words)

        elif self.analyzer == 'better':
            stop_words = self.get_stop_words()
            tokenize = self.xml_tokenizer

            return lambda doc: self.my_feature_extractor(
                tokenize(preprocess(self.decode(doc))), stop_words,
                self.ngram_range, lemmatize=self.lemmatize)

        else:
            raise ValueError('%s is not a valid tokenization scheme/analyzer' %
                             self.analyzer)


    def _term_count_dicts_to_matrix(self, term_count_dicts):
        """
        Converts a set ot document_term counts to a matrix; Input is a
        Counter(document_term-> frequency) object per document.
        Current version copied without functional modification from sklearn
        .feature_extraction.text.CountVectorizer
        """

        # how many tokens are there/ are unknown/ have been replaced
        total, unknown, replaced = 0, 0, 0

        if not hasattr(self, '_thesaurus'):
            #thesaurus has not been parsed yet
            if not self.thesaurus_file:
                # no thesaurus source, fall back to super behaviour
                logger.warn("F**k, no thesaurus!")
                return super(ThesaurusVectorizer,
                             self)._term_count_dicts_to_matrix(
                    term_count_dicts)
            else:
                # thesaurus file specified, parse it
                self._thesaurus = self.load_thesaurus()

        # sparse storage for document-term matrix (terminology note: term ==
        # feature)
        doc_id_indices = [] #which document the feature occurs in
        term_indices = []   #which term id appeared
        values = []         #values[i] = frequency(term[i]) in document[i]

        vocabulary = self.vocabulary_
        num_documents = 0
        logger.info("Building feature vectors, current vocab size is %d" %
                    len(vocabulary))

        for doc_id, term_count_dict in enumerate(term_count_dicts):
            num_documents += 1
            for document_term, count in term_count_dict.iteritems():
                total += 1
                term_index_in_vocab = vocabulary.get(document_term)
                if term_index_in_vocab is not None:
                #None if term is not in seen vocabulary
                    doc_id_indices.append(doc_id)
                    term_indices.append(term_index_in_vocab)
                    values.append(count)
                else: # this feature has not been seen before, replace it
                # the logger.info(below demonstrates that unseen words exist,)
                # i.e. vectorizer is not reducing the test set to the
                # training vocabulary
                # logger.info('Unknown token %s' % document_term)
                    unknown += 1
                    neighbours = self._thesaurus.get(document_term.lower())

                    # if there are any neighbours filter the list of
                    # neighbours so that it contains only pairs where
                    # the neighbour has been seen
                    neighbours = [(neighbour, sim) for neighbour, sim in
                                  neighbours[:self.k] if
                                  neighbour in self.vocabulary_] if neighbours else []
                    if len(neighbours) > 0:
                        replaced += 1
                    for neighbour, sim in neighbours:
                        logger.debug('Replacement. Doc %d: %s --> %s, '
                                     'sim = %f' % (
                                         doc_id, document_term, neighbour, sim))
                        inserted_feature_id = vocabulary.get(neighbour)
                        try:
                            position_in_lists = term_indices.index(
                                inserted_feature_id)
                            # the document already contain the feature we
                            # are about to insert into it, increment count
                            values[position_in_lists] += sim
                        except ValueError:
                            #this feature has not been inserted before
                            doc_id_indices.append(doc_id)
                            term_indices.append(vocabulary.get(neighbour))
                            values.append(sim)

            # free memory as we go
            term_count_dict.clear()

        # convert the three lists above to a numpy sparse matrix

        # this is sometimes a generator, convert to list to use len
        shape = (num_documents, max(vocabulary.itervalues()) + 1)
        spmatrix = sp.coo_matrix((values, (doc_id_indices, term_indices)),
                                 shape=shape, dtype=self.dtype)
        # remove frequencies if binary feature were requested
        if self.binary:
            spmatrix.data.fill(1)
        logger.info('Vectorizer: Data shape is %s' % (str(spmatrix.shape)))
        logger.info('Vectorizer: Total: %d Unknown: %d Replaced: %d' % (total,
                                                                        unknown,
                                                                        replaced))

        # temporarily store vocabulary
        f = './tmp_vocabulary'
        if self.log_vocabulary and not self.log_vocabulary_already:
            with open(f, 'w') as out:
                print '**************** writing vocab'
                pickle.dump(self.vocabulary_, out)
                self.log_vocabulary_already = True

        return spmatrix

    def xml_tokenizer(self, doc):
        tree = ET.fromstring(doc.encode("utf8"))
        tokens = [x.text for x in tree.iter('lemma')] if self.lemmatize else\
        [x.text for x in tree.iter('word')]

        if self.lowercase:
            tokens = [x.lower() for x in tokens]
        if self.use_pos:
            pos_tags = [x.text.upper() for x in tree.iter('pos')]
            if self.coarse_pos:
                pos_tags = self.coarsify(pos_tags)
            for i in range(len(pos_tags)):
                tokens[i] = '/'.join([tokens[i], pos_tags[i]])



        # if nothing else is specified, just return the words
        return tokens

    def coarsify(self, pos_tags):
        return [pos_coarsification_map[x.upper()] for x in pos_tags]


__author__ = 'mmb28'
# cache, to avoid re-loading all the time
thesauri = {}
corenlp = None
logger = _configure_logger()

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
})
