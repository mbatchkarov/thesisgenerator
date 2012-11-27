from collections import defaultdict
import scipy.sparse as sp
from sklearn.feature_extraction.text import  TfidfVectorizer

__author__ = 'mmb28'

# cache, to avoid re-loading all the time
thesauri = {}

def load_thesaurus(path):
    """
    Loads a Byblo-generated thesaurus form the specified file. If the file
    has been parsed already returns a cached version.
    """
    if not path:
        return None
    if thesauri.has_key(path):
        print 'Returning cached thesaurus for %s' % path
        return thesauri[path]
    else:
        print 'Loading thesaurus %s from disk' % path
        FILTERED = '___FILTERED___'.lower()
        neighbours = defaultdict(list)
        with open(path) as infile:
            for line in infile:
                tokens = line.strip().lower().split('\t')
                if len(tokens) % 2 == 0:#must have an even number of things
                    continue
                if tokens[0] != FILTERED:
                    neighbours[tokens[0]] = [(word, float(sim)) for (word, sim)
                                             in
                                             iterate_nonoverlapping_pairs(
                                                 tokens,
                                                 1)
                                             if word != FILTERED]
        thesauri[path] = neighbours
        return neighbours


def iterate_nonoverlapping_pairs(iterable, beg):
    for i in xrange(beg, len(iterable) - 1, 2): #step size 2
        yield (iterable[i], iterable[i + 1])


def my_feature_extractor(tokens, stop_words=None, ngram_range=(1, 1)):
    """
    Turn a document( a list of tokens) into a sequence of features. These
    include n-grams after stop words filtering,
    suffix features and shape features
    Based on sklearn.feature_extraction.text._word_ngrams
    """
    #todo add/enable feature functions here
    # handle stop words
    if stop_words is not None:
        tokens = [w for w in tokens if w not in stop_words]
        #
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

    return tokens # + last_chars + shapes


def my_analyzer():
    return lambda doc: my_feature_extractor(
        doc, None, None)


class ThesaurusVectorizer(TfidfVectorizer):
    """
    A thesaurus-backed CountVectorizer that replaces unknown features with
    their k nearest neighbours in the thesaurus
    """


    def __init__(self, thesaurus_file=None, k=1, sim_threshold=0.2,
                 input='content', charset='utf-8', charset_error='strict',
                 strip_accents=None, lowercase=True,
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
        #            if vocabulary:
        #                #  we only need the thesaurus if vocabulary is
        # fixed, i.e.
        #                #  if this is a test run
            self.thesaurus_file = thesaurus_file
            self._k = k
            self._sim_threshold = sim_threshold
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
            tokenize = self.build_tokenizer()

            return lambda doc: my_feature_extractor(
                tokenize(preprocess(self.decode(doc))), stop_words,
                self.ngram_range)

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
        if not hasattr(self, '_thesaurus'):
            #thesaurus has not been parsed yet
            if not self.thesaurus_file:
                # no thesaurus source, fall back to super behaviour
                print "F**k, no thesaurus!"
                return super(ThesaurusVectorizer,
                             self)._term_count_dicts_to_matrix(
                    term_count_dicts)
            else:
                # thesaurus file specified, parse it
                self._thesaurus = load_thesaurus(self.thesaurus_file)

        # sparse storage for document-term matrix (terminology note: term ==
        # feature)
        doc_id_indices = [] #which document the feature occurs in
        term_indices = []   #which term id appeared
        values = []         #values[i] = frequency(term[i]) in document[i]

        vocabulary = self.vocabulary_
        num_documents = 0
        print "Building feature vectors, current vocab size is %d" % len(
            vocabulary)

        for doc_id, term_count_dict in enumerate(term_count_dicts):
            num_documents += 1
            for document_term, count in term_count_dict.iteritems():
                term_index_in_vocab = vocabulary.get(document_term)
                if term_index_in_vocab is not None:
                #None if term is not in seen vocabulary
                    doc_id_indices.append(doc_id)
                    term_indices.append(term_index_in_vocab)
                    values.append(count)
                else: # this feature has not been seen before, replace it
                # the print below demonstrates that unseen words exist,
                # i.e. vectorizer is not reducing the test set to the
                # training vocabulary
                # print 'Unknown token %s' % document_term
                    neighbours = self._thesaurus.get(document_term)

                    # if there are any neighbours filter the list of
                    # neighbours so that it contains only pairs where
                    # the neighbour has been seen
                    neighbours = [(neighbour, sim) for neighbour, sim in
                                  neighbours[:self._k] if
                                  neighbour in self.vocabulary_ and sim >
                                  self._sim_threshold] if neighbours else []
                    for neighbour, sim in neighbours:
#                        print '***replacing %s with %s, sim = %f' % (
#                            document_term, neighbour, sim)
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
        print 'Data shape is ', spmatrix.shape
        return spmatrix

    def get_params(self, deep=True):
        out = super(ThesaurusVectorizer, self).get_params(deep)
        out['thesaurus'] = self.thesaurus_file
        out['sim_threshold'] = self._sim_threshold
        out['k'] = self._k
        return out

