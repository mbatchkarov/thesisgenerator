# coding=utf-8
from itertools import izip, tee
import logging
import pickle
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from thesisgenerator.classifiers import NoopTransformer
from thesisgenerator.plugins import tokenizers
from thesisgenerator.plugins.bov_feature_handlers import get_token_handler, get_stats_recorder
from thesisgenerator.utils.data_utils import get_named_object


class ThesaurusVectorizer(TfidfVectorizer):
    """
    A thesaurus-backed CountVectorizer that replaces unknown features with
    their k nearest neighbours in the thesaurus
    """

    def __init__(self, exp_name='', pipe_id=0, log_vocabulary=False,
                 lowercase=True,
                 input='content', charset='utf-8', charset_error='strict',
                 strip_accents=None,
                 preprocessor=None, tokenizer=None, analyzer='ngram',
                 stop_words=None, token_pattern=ur"(?u)\b\w\w+\b", min_n=None,
                 max_n=None, ngram_range=(1, 1), max_df=1.0, min_df=2,
                 max_features=None, vocabulary=None, binary=False, dtype=float,
                 norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False, use_tfidf=True,
                 record_stats=False, k=1,
                 sim_compressor='thesisgenerator.utils.misc.noop',
                 train_token_handler='thesisgenerator.plugins.bov_feature_handlers.BaseFeatureHandler',
                 decode_token_handler='thesisgenerator.plugins.bov_feature_handlers.BaseFeatureHandler',
                 train_thesaurus='',
                 decode_thesaurus=''):
        """
        Builds a vectorizer the way a TfidfVectorizer is built, and takes one
        extra param specifying the path the the Byblo-generated thesaurus.

        Do not do any real work here, this constructor is first invoked with
        no parameters by the pipeline and then all the right params are set
        through reflection. When __init__ terminates the class invariants
        have not been established, make sure to check& establish them in
        fit_transform()
        """
        self.log_vocabulary = log_vocabulary # if I should log the
        # vocabulary
        self.log_vocabulary_already = False # have I done it already
        self.use_tfidf = use_tfidf
        self.pipe_id = pipe_id
        self.exp_name = exp_name
        self.record_stats = record_stats
        self.k = k
        self.sim_compressor = sim_compressor
        self.train_token_handler = train_token_handler
        self.decode_token_handler = decode_token_handler

        # in case we want to mock the thesaurus (for unit testing or to
        # get a baseline). With this we can programatically build
        # "constant" thesauri that return the same for all possible
        # entries.
        # This access method should only be used at decode time,
        # so for now we just store it and rely on the vectorizer to
        # "activate" it after training is done
        self.train_thesaurus = train_thesaurus # a dict-like object
        self.decode_thesaurus = decode_thesaurus # a fully qualified name of a dict-like object,
        # instantiated through reflection
        self.thesaurus = train_thesaurus

        self.stats = None
        self.handler = None

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
                                                  dtype=dtype)

    def fit_transform(self, raw_documents, y=None):
        self.thesaurus = self.train_thesaurus
        self.handler = get_token_handler(self.train_token_handler,
                                         self.k,
                                         self.sim_compressor,
                                         self.train_thesaurus)
        self.stats = get_stats_recorder(self.record_stats)
        # a different stats recorder will be used for the testing data

        # self.try_to_set_vocabulary_from_thesaurus_keys()
        res = super(ThesaurusVectorizer, self).fit_transform(raw_documents, y)

        if self.decode_thesaurus:
            logging.warn('Will be using a different thesaurus through at decode time')
            self.thesaurus = get_named_object(self.decode_thesaurus)(self.vocabulary_)

        return res

    def transform(self, raw_documents):
        # record stats separately for the test set
        self.stats = get_stats_recorder(self.record_stats)

        self.handler = get_token_handler(self.decode_token_handler,
                                         self.k,
                                         self.sim_compressor,
                                         self.thesaurus)

        return super(ThesaurusVectorizer, self).transform(raw_documents)

    def pairwise(self, iterable):
        """
        s -> (s0,s1), (s1,s2), (s2, s3), ...

        From http://docs.python.org/2/library/itertools.html
        """

        a, b = tee(iterable)
        next(b, None)
        return izip(a, b)

    def my_feature_extractor(self, sentences, ngram_range=(1, 1)):
        """
        Turn a document( a list of tokens) into a sequence of features. These
        include n-grams after stop words filtering,
        suffix features and shape features
        Based on sklearn.feature_extraction.text._word_ngrams
        """

        # todo add/enable feature functions here
        #    last_chars = ['**suffix(%s)' % token[-1] for token in tokens]
        #    shapes = ['**shape(%s)' % "".join(
        #        'x' if l.islower() else '#' if l.isdigit()  else 'X' for l in
        # token)
        #              for token in
        #              tokens]

        features = []

        # extract sentence-internal token n-grams
        min_n, max_n = ngram_range
        for sentence in sentences:
            n_tokens = len(sentence)
            for n in xrange(min_n,
                            min(max_n + 1, n_tokens + 1)):
                for i in xrange(n_tokens - n + 1):
                    features.append(u" ".join(sentence[i: i + n]))
                    # extract sentence-internal adjective-noun compounds

        for sentence in sentences:
            for a, b in self.pairwise(sentence):
                if a.upper().endswith('/J') and b.upper().endswith('/N'):
                    features.append('{} {}'.format(a, b))

            return features  # + last_chars + shapes

    def my_analyzer(self):
        return lambda doc: self.my_feature_extractor(doc, None, None)

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
            tokenize = tokenizers.get_tokenizer()

            return lambda doc: self.my_feature_extractor(
                tokenize(preprocess(self.decode(doc))), self.ngram_range)

        elif self.analyzer == 'ngram':
            #assume input already tokenized
            return lambda token_list: self.my_feature_extractor(token_list, self.ngram_range)

        else:
            raise ValueError('%s is not a valid tokenization scheme/analyzer' %
                             self.analyzer)

    def _dump_vocabulary_for_debugging(self):
        # temporarily store vocabulary
        f = './tmp_vocabulary_%s_%d' % (self.exp_name, self.pipe_id)
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
        if hasattr(self, 'cv_number'):
            logging.info('cv_number=%s' % self.cv_number)
        logging.info('Converting features to vectors (with thesaurus lookup)')
        self._dump_vocabulary_for_debugging()

        if not self.use_tfidf:
            self._tfidf = NoopTransformer()

        logging.info('Using TF-IDF: %s, transformer is %s' % (self.use_tfidf,
                                                              self._tfidf))
        vocabulary = self.vocabulary_
        # thesaurus must ensure lookups for non-existent
        # keys return an empty list and not not raise an exception
        th = self.thesaurus

        # sparse storage for document-term matrix (terminology note: term ==
        # feature)
        doc_id_indices = [] # which document the feature occurs in
        term_indices = []   # which term id appeared
        values = []         # values[i] = frequency(term[i]) in document[i]

        num_documents = 0
        logging.debug("Building feature vectors, current vocab size is %d" %
                      len(vocabulary))

        for doc_id, term_count_dict in enumerate(term_count_dicts):
            num_documents += 1
            for document_term, count in term_count_dict.iteritems():

                term_index_in_vocab = vocabulary.get(document_term)
                is_in_vocabulary = term_index_in_vocab is not None
                # None if term is not in seen vocabulary

                try:
                    # some thesauri are defaultdict-s, must be queried with []
                    is_in_th = bool(th[document_term])
                except KeyError:
                    # some thesauri are plain dict-s and will raise a ValueError
                    is_in_th = False

                self.stats.register_token(document_term, is_in_vocabulary, is_in_th)

                params = (doc_id, doc_id_indices, document_term, term_indices,
                          term_index_in_vocab, values, count, vocabulary)
                if is_in_vocabulary and is_in_th:
                    self.handler.handle_IV_IT_feature(*params)
                if is_in_vocabulary and not is_in_th:
                    self.handler.handle_IV_OOT_feature(*params)
                if not is_in_vocabulary and is_in_th:
                    self.handler.handle_OOV_IT_feature(*params)
                if not is_in_vocabulary and not is_in_th:
                    self.handler.handle_OOV_OOT_feature(*params)
            term_count_dict.clear()

        # convert the three iterables above to a numpy sparse matrix
        # this is sometimes a generator, convert to list to use len
        shape = (num_documents, max(vocabulary.itervalues()) + 1)
        spmatrix = sp.coo_matrix((values, (doc_id_indices, term_indices)),
                                 shape=shape, dtype=self.dtype)

        # remove frequencies if binary feature were requested
        if self.binary:
            spmatrix.data.fill(1)
        logging.info('Vectorizer: Data shape is %s' % (str(spmatrix.shape)))
        self.stats.print_coverage_stats()
        logging.info('Done converting features to vectors')

        return spmatrix

