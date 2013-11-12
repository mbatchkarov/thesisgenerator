# coding=utf-8
from collections import defaultdict
import logging
import array
import scipy.sparse as sp
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from thesisgenerator.classifiers import NoopTransformer
from thesisgenerator.plugins import tokenizers
from thesisgenerator.plugins.bov_feature_handlers import get_token_handler, get_stats_recorder
from thesisgenerator.plugins.tokenizers import DocumentFeature


class ThesaurusVectorizer(TfidfVectorizer):
    """
    A thesaurus-backed CountVectorizer that replaces unknown features with
    their k nearest neighbours in the thesaurus
    """

    def __init__(self, exp_name='', pipe_id=0, lowercase=True,
                 input='content', encoding='utf-8', decode_error='strict',
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
                 vector_source=None):
        """
        Builds a vectorizer the way a TfidfVectorizer is built, and takes one
        extra param specifying the path the the Byblo-generated thesaurus.

        Do not do any real work here, this constructor is first invoked with
        no parameters by the pipeline and then all the right params are set
        through reflection. When __init__ terminates the class invariants
        have not been established, make sure to check& establish them in
        fit_transform()

        :param ngram_range: tuple(int,int), what n-grams to extract. If the range is (x,0), no n-ngrams of
        consecutive words will be extracted.
        """
        self.use_tfidf = use_tfidf
        self.pipe_id = pipe_id
        self.exp_name = exp_name
        self.record_stats = record_stats
        self.k = k
        self.sim_compressor = sim_compressor
        self.train_token_handler = train_token_handler
        self.decode_token_handler = decode_token_handler

        self.stats = None
        self.handler = None

        super(ThesaurusVectorizer, self).__init__(input=input,
                                                  encoding=encoding,
                                                  decode_error=decode_error,
                                                  strip_accents=strip_accents,
                                                  lowercase=lowercase,
                                                  preprocessor=preprocessor,
                                                  tokenizer=tokenizer,
                                                  analyzer=analyzer,
                                                  stop_words=stop_words,
                                                  token_pattern=token_pattern,
                                                  #min_n=min_n,
                                                  #max_n=max_n,
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

    def fit_transform(self, raw_documents, y=None, vector_source=None):
        self.vector_source = vector_source
        logging.debug('Identity of vector source is %d', id(vector_source))
        if vector_source:
            try:
                logging.debug('The BallTree is %s', vector_source.nbrs)
            except AttributeError:
                logging.debug('The vector source is %s', vector_source)

        self.handler = get_token_handler(self.train_token_handler,
                                         self.k,
                                         self.sim_compressor,
                                         self.vector_source)
        self.stats = get_stats_recorder(self.record_stats)
        # a different stats recorder will be used for the testing data

        res = super(ThesaurusVectorizer, self).fit_transform(raw_documents, y)

        # once training is done, convert all document features (unigrams and composable ngrams)
        # to a ditributional feature vector
        logging.info('Done vectorizing')
        return res, self.vocabulary_

    def transform(self, raw_documents):
        # record stats separately for the test set
        self.stats = get_stats_recorder(self.record_stats)

        self.handler = get_token_handler(self.decode_token_handler,
                                         self.k,
                                         self.sim_compressor,
                                         self.vector_source)

        # todo can't populate at this stage of the pipeline, because the vocabulary might
        # change if feature selection is enabled. Trying to do this will result in attempts to compose
        # features that we do not know how to compose because these have not been removed by FS
        #if self.vector_source:
        #    logging.info('Populating vector source %s prior to transform', self.vector_source)
        #    self.vector_source.populate_vector_space(self.vocabulary_.keys())
        return super(ThesaurusVectorizer, self).transform(raw_documents), self.vocabulary_

    def _extract_features_from_dependency_tree(self, features, parse_tree, token_indices):
        # extract sentence-internal adjective-noun compounds
        amods = [x[:2] for x in parse_tree.edges(data=True) if x[2]['type'] == 'amod']
        for head, dep in amods:
            features.append(DocumentFeature('AN', (token_indices[dep], token_indices[head])))

        # extract sentence-internal subject-verb-direct object compounds
        # todo how do we handle prepositional objects?
        verbs = [t.index for t in token_indices.values() if t.pos == 'V']
        subjects = set([(head, dep) for head, dep, opts in parse_tree.edges(data=True) if opts['type'] == 'nsubj'])
        objects = set([(head, dep) for head, dep, opts in parse_tree.edges(data=True) if opts['type'] == 'dobj'])
        subjverbobj = [(s[1], v, o[1]) for v in verbs for s in subjects for o in objects if s[0] == v and o[0] == v]
        for s, v, o in subjverbobj:
            features.append(DocumentFeature('SVO', (token_indices[s], token_indices[v], token_indices[o])))

    def my_feature_extractor(self, sentences, ngram_range=(1, 1)):
        """
        Turn a document( a list of tokens) into a sequence of features. These
        include n-grams after stop words filtering,
        suffix features and shape features
        Based on sklearn.feature_extraction.text._word_ngrams
        """
        features = []

        # extract sentence-internal token n-grams
        min_n, max_n = map(int, ngram_range)
        for sentence, (parse_tree, token_indices) in sentences:
            if parse_tree:
                self._extract_features_from_dependency_tree(features, parse_tree, token_indices)
            else:
                # sometimes an input document will have a sentence of one word, which has no dependencies
                # just ignore that and extract all the features that can be extracted without it
                logging.warn('Dependency tree not available')

            # extract sentence-internal n-grams
            if max_n > 0:
                n_tokens = len(sentence)
                for n in xrange(min_n, min(max_n + 1, n_tokens + 1)):
                    for i in xrange(n_tokens - n + 1):
                        features.append(DocumentFeature('%d-GRAM' % n, tuple(sentence[i: i + n])))


        #for sentence in sentences:
        #    for a, b in self._walk_pairwise(sentence):
        #        if a.upper().endswith('/J') and b.upper().endswith('/N'):
        #            features.append(('AN', (a, b)))

        #return self.vector_source.accept_features(features)  # + last_chars + shapes
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

        if self.analyzer == 'better':
            tokenize = tokenizers.get_tokenizer()
            return lambda doc: self.my_feature_extractor(
                tokenize(preprocess(self.decode(doc))), self.ngram_range)

        elif self.analyzer == 'ngram':
            #assume input already tokenized
            return lambda token_list: self.my_feature_extractor(token_list, self.ngram_range)
        else:
            return super(ThesaurusVectorizer, self).build_analyzer()


    def _count_vocab(self, raw_documents, fixed_vocab):
        """
        Modified from sklearn 0.14's CountVectorizer

        @params fixed_vocab True if the vocabulary attribute has been set, i.e. the vectorizer is trained
        """
        if hasattr(self, 'cv_number'):
            logging.info('cv_number=%s', self.cv_number)
        logging.info('Converting features to vectors (with thesaurus lookup)')

        if not self.use_tfidf:
            self._tfidf = NoopTransformer()
        logging.info('Using TF-IDF: %s, transformer is %s', self.use_tfidf, self._tfidf)

        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            # Add a new value when a new vocabulary item is seen, we're training now
            vocabulary = defaultdict(None)
            vocabulary.default_factory = vocabulary.__len__

        analyze = self.build_analyzer()
        j_indices = array.array(str("i"))
        indptr = array.array(str("i"))
        values = []
        indptr.append(0)
        for doc_id, doc in enumerate(raw_documents):
            for feature in analyze(doc):
                #####################  begin non-original code  #####################

                try:
                    feature_index_in_vocab = vocabulary[feature]
                except KeyError:
                    feature_index_in_vocab = None
                    # if term is not in seen vocabulary

                #is_in_vocabulary = bool(feature_index_in_vocab is not None)
                is_in_vocabulary = feature in vocabulary
                #is_in_th = bool(self.vector_source.get(feature))
                is_in_th = feature in self.vector_source if self.vector_source else False
                self.stats.register_token(feature, is_in_vocabulary, is_in_th)

                #j_indices.append(feature_index_in_vocab) # todo this is the original code, also updates vocabulary

                params = {'doc_id': doc_id, 'feature': feature,
                          'feature_index_in_vocab': feature_index_in_vocab,
                          'vocabulary': vocabulary, 'j_indices': j_indices, 'values': values}
                if is_in_vocabulary and is_in_th:
                    self.handler.handle_IV_IT_feature(**params)
                if is_in_vocabulary and not is_in_th:
                    self.handler.handle_IV_OOT_feature(**params)
                if not is_in_vocabulary and is_in_th:
                    self.handler.handle_OOV_IT_feature(**params)
                if not is_in_vocabulary and not is_in_th:
                    self.handler.handle_OOV_OOT_feature(**params)
                    #####################  end non-original code  #####################

                    #print doc_id, feature, len(j_indices)
            indptr.append(len(j_indices))

        if not fixed_vocab:
            # disable defaultdict behaviour
            vocabulary = dict(vocabulary)
            if not vocabulary:
                raise ValueError("empty vocabulary; perhaps the documents only"
                                 " contain stop words")

        # some Python/Scipy versions won't accept an array.array:
        if j_indices:
            j_indices = np.frombuffer(j_indices, dtype=np.intc)
        else:
            j_indices = np.array([], dtype=np.int32)
        indptr = np.frombuffer(indptr, dtype=np.intc)
        #values = np.ones(len(j_indices))

        X = sp.csr_matrix((values, j_indices, indptr),
                          shape=(len(indptr) - 1, len(vocabulary)),
                          dtype=self.dtype)
        X.sum_duplicates()  # nice that the summation is explicit
        self.stats.print_coverage_stats()
        return vocabulary, X

