from collections import defaultdict
import locale
import logging
import pickle
import traceback
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import NoopTransformer

preloaded_thesauri = {}


class ThesaurusVectorizer(TfidfVectorizer):
    """
    A thesaurus-backed CountVectorizer that replaces unknown features with
    their k nearest neighbours in the thesaurus
    """

    def __init__(self, thesaurus_files=None, k=1, sim_threshold=0.2,
                 normalise_entities=False,
                 lemmatize=False, log_vocabulary=False, coarse_pos=True,
                 input='content', charset='utf-8', charset_error='strict',
                 strip_accents=None, lowercase=True, use_pos=False,
                 preprocessor=None, tokenizer=None, analyzer='better',
                 stop_words=None, token_pattern=ur"(?u)\b\w\w+\b", min_n=None,
                 max_n=None, ngram_range=(1, 1), max_df=1.0, min_df=2,
                 max_features=None, vocabulary=None, binary=False, dtype=long,
                 norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False, use_tfidf=True, replace_all=False):
        """
        Builds a vectorizer the way a TfidfVectorizer is built, and takes one
        extra param specifying the path the the Byblo-generated thesaurus
        """
        try:
            self.normalise_entities = bool(normalise_entities)
            self.thesaurus_files = thesaurus_files
            self.k = int(k)
            self.sim_threshold = float(sim_threshold)
            self.log_vocabulary = log_vocabulary # if I should log the
            self.lemmatize = lemmatize
            self.use_pos = use_pos
            self.coarse_pos = coarse_pos
            # vocabulary
            self.log_vocabulary_already = False #have I done it already
            self.use_tfidf = use_tfidf
            self.replace_all = replace_all

            print thesaurus_files
            traceback.print_stack()

            # if not self.thesaurus_files:
            #     logging.getLogger('root').error("No thesaurus specified! "
            #                                     "One is required when "
            #                                     "using the replace_all "
            #                                     "option in order to set "
            #                                     "the dimensions of the "
            #                                     "semantic space")
            #     raise ValueError('Thesaurus is required')
            # else:
            #     # thesaurus file specified, parse it
            #     self._thesaurus = self.load_thesauri()

            # for parsing integers with comma for thousands separator
            locale.setlocale(locale.LC_ALL, 'en_US')
        except KeyError:
            pass

        # vocabulary_to_pass_in = vocabulary if self.replace_all else \
        #     self._thesaurus.keys()
        vocabulary_to_pass_in = vocabulary

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
                                                  vocabulary=vocabulary_to_pass_in,
                                                  use_idf=use_idf,
                                                  smooth_idf=smooth_idf,
                                                  sublinear_tf=sublinear_tf,
                                                  binary=binary,
                                                  norm=norm,
                                                  dtype=dtype
        )

    def load_thesauri(self):
        """
        Loads a set Byblo-generated thesaurus form the specified file and
        returns their union. If any of the files has been parsed already a
        cached version is used.

        Parameters:
        self.thesaurus_files: string, path the the Byblo-generated thesaurus
        self.use_pos: boolean, whether the PoS tags should be stripped from
        entities (if they are present)
        self.sim_threshold: what is the min similarity for neighbours that
        should be loaded
        """
        if not self.thesaurus_files:
            return None

        result = {}
        logging.getLogger('main').debug(self.thesaurus_files)
        for path in self.thesaurus_files:
            if path in preloaded_thesauri:
                logging.getLogger('main').info('Returning cached thesaurus '
                                               'for %s' % path)
                result.update(preloaded_thesauri[path])
            else:
                logging.getLogger('main').info(
                    'Loading thesaurus %s from disk' % path)
                logging.getLogger('main').debug(
                    'PoS: %r, coarse: %r, threshold %r, k=%r' %
                    (self.use_pos,
                     self.coarse_pos,
                     self.sim_threshold,
                     self.k))
                FILTERED = '___FILTERED___'.lower()
                curr_thesaurus = defaultdict(list)
                with open(path) as infile:
                    for line in infile:
                        tokens = line.strip().split('\t')
                        if len(tokens) % 2 == 0:
                        #must have an odd number of things, one for the entry and
                        # pairs for (neighbour, similarity)
                            continue
                        if tokens[0] != FILTERED:
                            to_insert = [(word.lower(), float(sim)) for
                                         (word, sim)
                                         in
                                         self.iterate_nonoverlapping_pairs(
                                             tokens, 1, self.k)
                                         if
                                         word != FILTERED and sim >
                                         self.sim_threshold]
                            # the step above may filter out all neighbours of an
                            # entry. if this happens, do not bother adding it
                            if len(to_insert) > 0:
                                if tokens[0] in curr_thesaurus:
                                    logging.getLogger('main').debug(
                                        'Multiple entries for "%s" found' %
                                        tokens[0])
                                curr_thesaurus[tokens[0].lower()].extend(
                                    to_insert)

                # note- do not attempt to lowercase if the thesaurus has not already been
                # lowercased- may result in multiple neighbour lists for the same entry
                logging.getLogger('main').info('Caching thesaurus %s' % path)
                preloaded_thesauri[path] = curr_thesaurus
                result.update(curr_thesaurus)

        logging.getLogger('main').info(
            'Thesaurus contains %d entries' % len(result))
        logging.getLogger('main').debug(
            'Thesaurus sample %r' % result.items()[:2])
        return result


    def iterate_nonoverlapping_pairs(self, iterable, beg, end):
        for i in xrange(beg, min(len(iterable) - 1, 2 * end), 2):  #step size 2
            yield (iterable[i], iterable[i + 1])


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
        logging.getLogger('main').info(
            'Building and starting analysis (tokenize, stopw, feature extract')
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
        logging.getLogger('main').info(
            'Converting features to vectors (with thesaurus lookup)')

        if not self.use_tfidf:
            self._tfidf = NoopTransformer()

        logging.getLogger('main').info(
            'Using TF-IDF: %s, transformer is %s' % (self.use_tfidf,
                                                     self._tfidf))

        if not hasattr(self, '_thesaurus'):
            #thesaurus has not been parsed yet
            if not self.thesaurus_files:
                # no thesaurus source, fall back to super behaviour
                logging.getLogger('main').warn("F**k, no thesaurus!")
                return super(ThesaurusVectorizer,
                             self)._term_count_dicts_to_matrix(term_count_dicts)
            else:
                # thesaurus file specified, parse it
                self._thesaurus = self.load_thesauri()

        # sparse storage for document-term matrix (terminology note: term ==
        # feature)
        doc_id_indices = [] #which document the feature occurs in
        term_indices = []   #which term id appeared
        values = []         #values[i] = frequency(term[i]) in document[i]

        vocabulary = self.vocabulary_
        num_documents = 0
        logging.getLogger('main').debug(
            "Building feature vectors, current vocab size is %d" %
            len(vocabulary))

        # how many tokens are there/ are unknown/ have been replaced
        num_tokens, unknown_tokens, found_tokens, replaced_tokens = 0, 0, 0, 0
        known_tokens = 0
        all_types = set()
        unknown_types = set()
        found_types = set()
        replaced_types = set()
        known_types = set()
        for doc_id, term_count_dict in enumerate(term_count_dicts):
            num_documents += 1
            for document_term, count in term_count_dict.iteritems():
                all_types.add(document_term)
                num_tokens += 1
                term_index_in_vocab = vocabulary.get(document_term)
                if term_index_in_vocab is not None:
                #None if term is not in seen vocabulary
                    # logging.getLogger('main').debug(
                    #     'Known token in doc %d: %s' % (doc_id, document_term))
                    known_tokens += 1
                    known_types.add(document_term)
                    doc_id_indices.append(doc_id)
                    term_indices.append(term_index_in_vocab)
                    values.append(count)

                else:
                # this feature has not been seen before, replace it
                # the logger.info(below demonstrates that unseen words exist,)
                # i.e. vectorizer is not reducing the test set to the
                # training vocabulary
                    unknown_tokens += 1
                    unknown_types.add(document_term)
                    logging.getLogger('main').debug(
                        'Unknown token in doc %d: %s' % (doc_id, document_term))

                    neighbours = self._thesaurus.get(document_term)

                    # if there are any neighbours filter the list of
                    # neighbours so that it contains only pairs where
                    # the neighbour has been seen
                    if neighbours:
                        found_tokens += 1
                        found_types.add(document_term)
                        logging.getLogger('main').debug('Found thesaurus entry '
                                                        'for %s' % document_term)
                    neighbours = [(neighbour, sim) for neighbour, sim in
                                  neighbours[:self.k] if
                                  neighbour in self.vocabulary_] if neighbours else []
                    if len(neighbours) > 0:
                        replaced_tokens += 1
                        replaced_types.add(document_term)
                    for neighbour, sim in neighbours:
                        logging.getLogger('main').debug(
                            'Replacement. Doc %d: %s --> %s, '
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
        logging.getLogger('main').debug(
            'Vectorizer: Data shape is %s' % (str(spmatrix.shape)))
        logging.getLogger('main').info(
            'Vectorizer: '
            'Total tokens: %d, '
            'Unknown tokens: %d,  Found tokens: %d, Replaced tokens: %d, '
            'Total types: %d, '
            'Unknown types: %d,  Found types: %d, Replaced types: %d, '
            'Known tokens: %d, Known types: %d' % (
                num_tokens,
                unknown_tokens, found_tokens, replaced_tokens,
                len(all_types),
                len(unknown_types), len(found_types), len(replaced_types),
                known_tokens, len(known_types)))

        # temporarily store vocabulary
        f = './tmp_vocabulary'
        if self.log_vocabulary and not self.log_vocabulary_already:
            with open(f, 'w') as out:
                logging.getLogger('main').info('Writing debug info')
                pickle.dump(self.vocabulary_, out)
                self.log_vocabulary_already = True

        logging.getLogger('main').info('Done converting features to vectors')

        return spmatrix

    def xml_tokenizer(self, doc):
        """
        Tokenizes a Stanford Core NLP processed document by parsing the XML and
        extracting tokens and their lemmas, with optional lowercasing
        If requested, the named entities will be replaced with the respective
         type, e.g. PERSON or ORG, otherwise numbers and punctuation will be
         canonicalised
        """

        try:
            import xml.etree.cElementTree as ET
        except ImportError:
            logging.getLogger('main').warn('cElementTree not available')
            import xml.etree.ElementTree as ET

        try:
            tree = ET.fromstring(doc.encode("utf8"))
            tokens = []
            for element in tree.findall('.//token'):
                if self.lemmatize:
                    txt = element.find('lemma').text
                else:
                    txt = element.find('word').text

                if self.lowercase:
                    txt = txt.lower()

                # check if the token is a number before things have been done
                #  to it
                am_i_a_number = is_number(txt)

                pos = element.find('pos').text.upper()
                if self.use_pos:
                    if self.coarse_pos:
                        pos = pos_coarsification_map[
                            pos.upper()]
                    txt = '%s/%s' % (txt, pos)

                if self.normalise_entities:
                    try:
                        iob_tag = element.find('ner').text.upper()
                    except AttributeError:
                        logging.getLogger('main').error(
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
                logging.getLogger('tokens').debug(txt)
                tokens.append(txt)

        except ET.ParseError:
            pass
            # on OSX the .DS_Store file is passed in, if it exists
            # just ignore it
        return tokens


def is_number(s):
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
