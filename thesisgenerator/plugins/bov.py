# coding=utf-8
from collections import defaultdict
import logging
import array
import numbers
import networkx as nx
from operator import attrgetter
import scipy.sparse as sp
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from thesisgenerator.classifiers import NoopTransformer
from thesisgenerator.plugins.bov_feature_handlers import get_token_handler
from discoutils.tokens import DocumentFeature
from thesisgenerator.plugins.stats import get_stats_recorder


class ThesaurusVectorizer(TfidfVectorizer):
    """
    A thesaurus-backed CountVectorizer that replaces unknown features with
    their k nearest neighbours in the thesaurus
    """

    def __init__(self, lowercase=True,
                 input='content', encoding='utf-8', decode_error='strict',
                 strip_accents=None,
                 preprocessor=None, tokenizer=None, analyzer='ngram',
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 max_df=1.0, min_df=0,
                 max_features=None, vocabulary=None, binary=False, dtype=float,
                 norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False, use_tfidf=True,
                 record_stats=True, k=1,
                 sim_compressor='thesisgenerator.utils.misc.unit',
                 train_token_handler='thesisgenerator.plugins.bov_feature_handlers.BaseFeatureHandler',
                 decode_token_handler='thesisgenerator.plugins.bov_feature_handlers.BaseFeatureHandler',
                 train_time_opts={'extract_unigram_features': ['J', 'N'],
                                  'extract_phrase_features': ['AN', 'NN', 'VO', 'SVO']},
                 decode_time_opts={'extract_unigram_features': '',
                                   'extract_phrase_features': ['AN', 'NN']},
                 standard_ngram_features=0,
                 remove_features_with_NER=False,
                 random_neighbour_thesaurus=False, **kwargs
                 ):
        """
        Builds a vectorizer the way a TfidfVectorizer is built, and takes one
        extra param specifying the path the the Byblo-generated thesaurus.

        Do not do any real work here, this constructor is first invoked with
        no parameters by the pipeline and then all the right params are set
        through reflection. When __init__ terminates the class invariants
        have not been established, make sure to check& establish them in
        fit_transform()

        :param standard_ngram_features: int. Extract standard (adjacent) ngram features up to this length
        """
        self.use_tfidf = use_tfidf
        self.record_stats = record_stats
        self.k = k
        self.sim_compressor = sim_compressor
        self.train_token_handler = train_token_handler
        self.decode_token_handler = decode_token_handler
        self.train_time_opts = train_time_opts
        self.decode_time_opts = decode_time_opts
        self.remove_features_with_NER = remove_features_with_NER
        self.random_neighbour_thesaurus = random_neighbour_thesaurus
        self.standard_ngram_features = standard_ngram_features

        self.stats = None
        self.handler = None
        self.entity_ner_tags = {'ORGANIZATION', 'PERSON', 'LOCATION'}

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

    def fit_transform(self, raw_documents, y=None, vector_source=None, stats_hdf_file=None, cv_fold=-1):
        self.cv_fold = cv_fold
        self.extract_unigram_features = self.train_time_opts['extract_unigram_features']
        self.extract_phrase_features = self.train_time_opts['extract_phrase_features']
        self.thesaurus = vector_source
        self.handler = get_token_handler(self.train_token_handler,
                                         self.k,
                                         self.sim_compressor,
                                         self.thesaurus)
        # requested stats that to go HDF, store the name so we can record stats to that name at decode time too
        self.stats_hdf_file_ = stats_hdf_file
        self.stats = get_stats_recorder(self.record_stats, stats_hdf_file, 'tr', cv_fold, self.k)
        # a different stats recorder will be used for the testing data

        # ########## BEGIN super.fit_transform ##########
        # this is a modified version of super.fit_transform which works with an empty vocabulary
        self._validate_vocabulary()
        max_df = self.max_df
        min_df = self.min_df
        max_features = self.max_features

        vocabulary, X = self._count_vocab(raw_documents, self.fixed_vocabulary_)
        X = X.tocsc()

        if self.binary:
            X.data.fill(1)

        if not self.fixed_vocabulary_:
            if vocabulary:
                X = self._sort_features(X, vocabulary)

                n_doc = X.shape[0]
                max_doc_count = (max_df
                                 if isinstance(max_df, numbers.Integral)
                                 else int(round(max_df * n_doc)))
                min_doc_count = (min_df
                                 if isinstance(min_df, numbers.Integral)
                                 else int(round(min_df * n_doc)))
                if max_doc_count < min_doc_count:
                    raise ValueError(
                        "max_df corresponds to < documents than min_df")
                X, self.stop_words_ = self._limit_features(X, vocabulary,
                                                           max_doc_count,
                                                           min_doc_count,
                                                           max_features)

            self.vocabulary_ = vocabulary
        # ######### END super.fit_transform ##########
        if (self.thesaurus and hasattr(self.thesaurus, 'get_nearest_neighbours') and
                hasattr(self.thesaurus.get_nearest_neighbours, 'cache_info')):
            logging.info('NN cache info: %s', self.thesaurus.get_nearest_neighbours.cache_info())
        logging.info('Matrix shape is %r after vectorization', X.shape)
        return X, self.vocabulary_

    def transform(self, raw_documents):
        self.extract_unigram_features = self.decode_time_opts['extract_unigram_features']
        self.extract_phrase_features = self.decode_time_opts['extract_phrase_features']
        if not hasattr(self, 'vocabulary_'):
            self._check_vocabulary()

        if not hasattr(self, 'vocabulary_') or len(self.vocabulary_) == 0:
            raise ValueError("Vocabulary wasn't fitted or is empty!")
        # record stats separately for the test set
        self.stats = get_stats_recorder(self.record_stats, self.stats_hdf_file_, 'ev',
                                        self.cv_fold, self.k)

        if self.random_neighbour_thesaurus:
            # this is a bit of hack and a waste of effort, since a thesaurus will have been loaded first
            logging.info('Building random neighbour vector source with vocabulary of size %d', len(self.vocabulary_))
            self.thesaurus.k = self.k
            self.thesaurus.vocab = list(self.vocabulary_.keys())

        self.handler = get_token_handler(self.decode_token_handler,
                                         self.k,
                                         self.sim_compressor,
                                         self.thesaurus)

        # todo can't populate at this stage of the pipeline, because the vocabulary might
        # change if feature selection is enabled. Trying to do this will result in attempts to compose
        # features that we do not know how to compose because these have not been removed by FS
        # if self.thesaurus:
        # logging.info('Populating vector source %s prior to transform', self.thesaurus)
        # self.thesaurus.populate_vector_space(self.vocabulary_.keys())

        # BEGIN a modified version of super.transform that works when vocabulary is empty
        _, X = self._count_vocab(raw_documents, fixed_vocab=True)
        if self.binary:
            X.data.fill(1)
            # END super.transform

        try:
            # try and close the shelf
            self.d.close()
        except Exception:
            # may not be a shelf after all
            pass
        if (self.thesaurus and hasattr(self.thesaurus, 'get_nearest_neighbours') and
                hasattr(self.thesaurus.get_nearest_neighbours, 'cache_info')):
            logging.info('NN cache info: %s', self.thesaurus.get_nearest_neighbours.cache_info())
        return X, self.vocabulary_

    def _remove_features_containing_named_entities(self, features):
        return [f for f in features if not any(token.ner in self.entity_ner_tags for token in f.tokens)]

    def extract_features_from_dependency_tree(self, parse_tree):
        new_features = []

        # extract sentence-internal adjective-noun compounds
        if 'AN' in self.extract_phrase_features:
            # get tuples of (head, dependent) for each amod relation in the tree
            # also enforce that head is a noun, dependent is an adjective
            for head, dep, data in parse_tree.edges(data=True):
                if data['type'] == 'amod' and head.pos == 'N' and dep.pos == 'J':
                    new_features.append(DocumentFeature('AN', (dep, head)))

        if 'VO' in self.extract_phrase_features or 'SVO' in self.extract_phrase_features:
            # extract sentence-internal subject-verb-direct object compounds
            # todo how do we handle prepositional objects?
            verbs = [t for t in parse_tree.nodes() if t.pos == 'V']

            objects = set([(head, dep) for head, dep, data in parse_tree.edges(data=True)
                           if data['type'] == 'dobj' and head.pos == 'V' and dep.pos == 'N'])

        if 'SVO' in self.extract_phrase_features:
            subjects = set([(head, dep) for head, dep, opts in parse_tree.edges(data=True) if
                            opts['type'] == 'nsubj' and head.pos == 'V' and dep.pos == 'N'])

            subjverbobj = [(s[1], v, o[1]) for v in verbs for s in subjects for o in objects if s[0] == v and o[0] == v]

            for s, v, o in subjverbobj:
                new_features.append(DocumentFeature('SVO', (s, v, o)))

        if 'SVO' in self.extract_phrase_features:
            verbobj = [(v, o[1]) for v in verbs for o in objects if o[0] == v]
            for v, o in verbobj:
                new_features.append(DocumentFeature('VO', (v, o)))

        if 'NN' in self.extract_phrase_features:
            for head, dep, data in parse_tree.edges(data=True):
                if data['type'] == 'nn' and head.pos == 'N' and dep.pos == 'N':
                    new_features.append(DocumentFeature('NN', (dep, head)))

        if self.remove_features_with_NER:
            return self._remove_features_containing_named_entities(new_features)
        return new_features

    def extract_features_from_token_list(self, doc_sentences):
        """
        Turn a document( a list of tokens) into a sequence of features.
        """
        features = []

        # extract sentence-internal token n-grams

        for parse_tree in doc_sentences:
            if not parse_tree:  # the sentence segmenter sometimes returns empty sentences
                continue

            if parse_tree:
                features.extend(self.extract_features_from_dependency_tree(parse_tree))
            else:
                # sometimes an input document will have a sentence of one word, which has no dependencies
                # just ignore that and extract all the features that can be extracted without it
                logging.warning('Dependency tree not available')

            # extract sentence-internal n-grams of the right PoS tag
            if self.extract_unigram_features:
                # just unigrams, can get away without sorting the tokens
                for token in parse_tree.nodes_iter():
                    if token.pos not in self.extract_unigram_features:
                        continue
                    features.append(DocumentFeature('1-GRAM', (token,)))

            # some tests use standard bigrams, extract them too
            if self.standard_ngram_features > 1:
                # the tokens are stored as nodes in the parse tree in ANY order, sort them
                sentence = sorted(parse_tree.nodes(), key=attrgetter('index'))
                n_tokens = len(sentence)
                for n in range(2, min(self.standard_ngram_features + 1, n_tokens + 1)):
                    for i in range(n_tokens - n + 1):
                        feature = DocumentFeature('%d-GRAM' % n, tuple(sentence[i: i + n]))
                        features.append(feature)
        # it doesn't matter where in the sentence/document these features were found
        # erase their index
        for feature in features:
            for token in feature.tokens:
                token.index = 'any'

        # remove all features that aren't right- they are there because the code above doesnt
        # put the features through the validation code in DocumentFeature.from_string
        # e.g. the verb phrase "123/V_$$$/N" is not put through validation, so it will be returned as feature
        return [f for f in features if DocumentFeature.from_string(str(f)).type != 'EMPTY']

    def filter_preextracted_features(self, feature_list):
        res = []
        for feat_str in feature_list:
            feat = DocumentFeature.from_string(feat_str)
            if feat.type == 'EMPTY':
                continue
            if feat.type == '1-GRAM' and feat.tokens[0].pos not in self.extract_unigram_features:
                continue
            if feat.type != '1-GRAM' and feat.type not in self.extract_phrase_features:
                continue
            res.append(feat)
        # logging.info('Had %d feats, keeping %d', len(feature_list), len(res))
        return res

    def build_analyzer(self):
        """
        Return a callable that handles preprocessing,
        tokenization and any additional feature extraction.
        """
        if hasattr(self.analyzer, '__call__'):
            return self.analyzer

        def _extract_or_filter(thing):
            if isinstance(thing[0], str):
                # assume input already feature-extracted
                return self.filter_preextracted_features(thing)
            if isinstance(thing[0], nx.DiGraph):
                # assume input tokenized, but features have not been extracted
                # corpora used in unit tests are like this
                return self.extract_features_from_token_list(thing)

        return _extract_or_filter

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
            if doc_id % 1000 == 0:
                logging.info('Done %d/%d documents...', doc_id, len(raw_documents))
            for feature in analyze(doc):
                # ####################  begin non-original code  #####################
                self._process_single_feature(feature, j_indices, values, vocabulary)
            indptr.append(len(j_indices))

        if not fixed_vocab:
            # disable defaultdict behaviour
            vocabulary = dict(vocabulary)
            if not vocabulary:
                logging.error('Empty vocabulary')
        # some Python/Scipy versions won't accept an array.array:
        if j_indices:
            j_indices = np.frombuffer(j_indices, dtype=np.intc)
        else:
            j_indices = np.array([], dtype=np.int32)
        indptr = np.frombuffer(indptr, dtype=np.intc)
        # values = np.ones(len(j_indices))

        X = sp.csr_matrix((values, j_indices, indptr),
                          shape=(len(indptr) - 1, len(vocabulary)),
                          dtype=self.dtype)
        X.sum_duplicates()  # nice that the summation is explicit
        self.stats.consolidate_stats()
        return vocabulary, X

    def _process_single_feature(self, feature, j_indices, values, vocabulary):
        try:
            feature_index_in_vocab = vocabulary[feature]
        except KeyError:
            feature_index_in_vocab = None
            # if term is not in seen vocabulary
        # is_in_vocabulary = bool(feature_index_in_vocab is not None)
        is_in_vocabulary = feature in vocabulary
        # is_in_th = bool(self.thesaurus.get(feature))
        is_in_th = feature in self.thesaurus if self.thesaurus else False
        self.stats.register_token(feature, is_in_vocabulary, is_in_th)
        # j_indices.append(feature_index_in_vocab) # todo this is the original code, also updates vocabulary
        params = {'feature': feature,
                  'feature_index_in_vocab': feature_index_in_vocab,
                  'vocabulary': vocabulary, 'j_indices': j_indices,
                  'values': values, 'stats': self.stats}
        if is_in_vocabulary and is_in_th:
            self.handler.handle_IV_IT_feature(**params)
        if is_in_vocabulary and not is_in_th:
            self.handler.handle_IV_OOT_feature(**params)
        if not is_in_vocabulary and is_in_th:
            self.handler.handle_OOV_IT_feature(**params)
        if not is_in_vocabulary and not is_in_th:
            self.handler.handle_OOV_OOT_feature(**params)
            #####################  end non-original code  #####################
