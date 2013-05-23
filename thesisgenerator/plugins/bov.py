# coding=utf-8
import locale
import logging
import pickle

import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from thesisgenerator.utils import NoopTransformer

from thesisgenerator.plugins.thesaurus_loader import load_thesauri
from thesisgenerator.plugins.tokenizers import xml_tokenizer


class ThesaurusVectorizer(TfidfVectorizer):
    """
    A thesaurus-backed CountVectorizer that replaces unknown features with
    their k nearest neighbours in the thesaurus
    """

    def __init__(self, pipe_id=0, log_vocabulary=False, lowercase=True,
                 input='content', charset='utf-8', charset_error='strict',
                 strip_accents=None,
                 preprocessor=None, tokenizer=None, analyzer='better',
                 stop_words=None, token_pattern=ur"(?u)\b\w\w+\b", min_n=None,
                 max_n=None, ngram_range=(1, 1), max_df=1.0, min_df=2,
                 max_features=None, vocabulary=None, binary=False, dtype=float,
                 norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False, use_tfidf=True, replace_all=False):
        """
        Builds a vectorizer the way a TfidfVectorizer is built, and takes one
        extra param specifying the path the the Byblo-generated thesaurus
        """
        try:
            self.log_vocabulary = log_vocabulary # if I should log the
            # vocabulary
            self.log_vocabulary_already = False #have I done it already
            self.use_tfidf = use_tfidf
            self.replace_all = bool(replace_all)
            self.pipe_id = pipe_id
            # for parsing integers with comma for thousands separator
            locale.setlocale(locale.LC_ALL, 'en_US')
        except KeyError:
            pass

        super(ThesaurusVectorizer, self).__init__(input=input,
                                                  charset=charset,
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
                                                  max_df=max_df,
                                                  min_df=min_df,
                                                  max_features=max_features,
                                                  vocabulary=vocabulary,
                                                  use_idf=use_idf,
                                                  smooth_idf=smooth_idf,
                                                  sublinear_tf=sublinear_tf,
                                                  binary=binary,
                                                  norm=norm,
                                                  dtype=dtype
        )

    def fit_transform(self, raw_documents, y=None):
        self._thesaurus = load_thesauri()
        if self.replace_all:
            if not self._thesaurus:
                raise ValueError('A thesaurus is required when using '
                                 'replace_all')
            self.vocabulary_ = {k: v for v, k in
                                enumerate(sorted(self._thesaurus.keys()))}
            self.fixed_vocabulary = True
        return super(ThesaurusVectorizer, self).fit_transform(raw_documents,
                                                              y)

    def fit(self, X, y=None, **fit_params):
        self._thesaurus = load_thesauri()
        if self.replace_all:
            if not self._thesaurus:
                raise ValueError('A thesaurus is required when using '
                                 'replace_all')
            self.vocabulary_ = {k: v for v, k in
                                enumerate(sorted(self._thesaurus.keys()))}
            self.fixed_vocabulary = True
        return super(ThesaurusVectorizer, self).fit(X, y, **fit_params)

    def my_feature_extractor(self, tokens, stop_words=None,
                             ngram_range=(1, 1)):
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
            tokens = [w for w in tokens if w.lower() not in stop_words and
                                           len(w) > 3]

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
        logging.getLogger('root').info(
            'Building and starting analysis (tokenize, stopw, feature extract)')
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
            tokenize = xml_tokenizer

            return lambda doc: self.my_feature_extractor(
                tokenize(preprocess(self.decode(doc))), stop_words,
                self.ngram_range)

        else:
            raise ValueError('%s is not a valid tokenization scheme/analyzer' %
                             self.analyzer)

    def _dump_vocabulary_for_debugging(self):
        # temporarily store vocabulary
        f = './tmp_vocabulary%d' % self.pipe_id
        if self.log_vocabulary and not self.log_vocabulary_already:
            with open(f, 'w') as out:
                pickle.dump(self.vocabulary_, out)
                self.log_vocabulary_already = True

    def _term_count_dicts_to_matrix(self, term_count_dicts):
        """
        Converts a set ot document_term counts to a matrix; Input is a
        Counter(document_term-> frequency) object per document.
        Current version copied without functional modification from sklearn
        .feature_extraction.text.CountVectorizer
        """
        logging.getLogger('root').info(
            'Converting features to vectors (with thesaurus lookup)')
        self._dump_vocabulary_for_debugging()

        if not self.use_tfidf:
            self._tfidf = NoopTransformer()

        logging.getLogger('root').info(
            'Using TF-IDF: %s, transformer is %s' % (self.use_tfidf,
                                                     self._tfidf))

        if not self._thesaurus:
            # no thesaurus was loaded in the constructor,
            # fall back to super behaviour
            logging.getLogger('root').warn("F**k, no thesaurus!")
            return super(ThesaurusVectorizer,
                         self)._term_count_dicts_to_matrix(term_count_dicts)

        # sparse storage for document-term matrix (terminology note: term ==
        # feature)
        doc_id_indices = [] #which document the feature occurs in
        term_indices = []   #which term id appeared
        values = []         #values[i] = frequency(term[i]) in document[i]

        vocabulary = self.vocabulary_
        num_documents = 0
        logging.getLogger('root').debug(
            "Building feature vectors, current vocab size is %d" %
            len(vocabulary))

        # how many tokens are there/ are unknown/ have been replaced
        num_tokens, unknown_tokens, found_tokens, replaced_tokens = 0, 0, 0, 0
        all_types = set()
        unknown_types = set()
        found_types = set()
        replaced_types = set()
        for doc_id, term_count_dict in enumerate(term_count_dicts):
            num_documents += 1
            for document_term, count in term_count_dict.iteritems():
                all_types.add(document_term)
                num_tokens += 1
                term_index_in_vocab = vocabulary.get(document_term)
                in_vocab = term_index_in_vocab is not None #None if term is not in seen vocabulary

                if in_vocab and not self.replace_all:
                # insert the term itself as a feature
                    logging.getLogger('root').debug(
                        'Known token in doc %d: %s' % (
                            doc_id, document_term))
                    doc_id_indices.append(doc_id)
                    term_indices.append(term_index_in_vocab)
                    values.append(count)
                elif not in_vocab and self.replace_all:
                    logging.getLogger('root').debug(
                        'Non-thesaurus token in doc %d: %s' % (
                            doc_id, document_term))
                    # unknown term, but it's not in thesaurus; nothing
                    # we can do about it
                else:
                # replace term with its k nearest neighbours from the thesaurus

                #  logger.info below demonstrates that unseen words exist,
                # i.e. vectorizer is not reducing the tests set to the
                # training vocabulary
                    unknown_tokens += 1
                    unknown_types.add(document_term)
                    logging.getLogger('root').debug(
                        'Unknown token in doc %d: %s' % (doc_id, document_term))

                    neighbours = self._thesaurus.get(document_term)

                    # if there are any neighbours filter the list of
                    # neighbours so that it contains only pairs where
                    # the neighbour has been seen
                    if neighbours:
                        found_tokens += 1
                        found_types.add(document_term)
                        logging.getLogger('root').debug('Found thesaurus entry '
                                                        'for %s' % document_term)
                    neighbours = [(neighbour, sim) for neighbour, sim in
                                  neighbours if
                                  neighbour in self.vocabulary_] if neighbours \
                        else []
                    if len(neighbours) > 0:
                        replaced_tokens += 1
                        replaced_types.add(document_term)
                    for neighbour, sim in neighbours:
                        logging.getLogger('root').debug(
                            'Replacement. Doc %d: %s --> %s, '
                            'sim = %f' % (
                                doc_id, document_term, neighbour, sim))
                        # todo the document may already contain the feature we
                        # are about to insert into it,
                        # a mergin strategy is required,
                        # e.g. what do we do if the document has the word X
                        # in it and we encounter X again. By default,
                        # scipy uses addition
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
        logging.getLogger('root').debug(
            'Vectorizer: Data shape is %s' % (str(spmatrix.shape)))
        logging.getLogger('root').info(
            'Vectorizer: Total tokens: %d, Unknown tokens: %d,  Found tokens: %d,'
            ' Replaced tokens: %d, Total types: %d, Unknown types: %d,  '
            'Found types: %d, Replaced types: %d' % (
                num_tokens, unknown_tokens, found_tokens, replaced_tokens,
                len(all_types), len(unknown_types), len(found_types),
                len(replaced_types)))
        logging.getLogger('root').info('Done converting features to vectors')

        return spmatrix

