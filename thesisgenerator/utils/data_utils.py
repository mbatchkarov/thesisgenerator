import logging
import os
from discoutils.misc import ContainsEverything
from joblib import Memory
import random
from sklearn.datasets import load_files
from thesisgenerator.classifiers import NoopTransformer
from thesisgenerator.composers.vectorstore import UnigramVectorSource, CompositeVectorSource, \
    PrecomputedSimilaritiesVectorSource
from thesisgenerator.plugins import tokenizers
from thesisgenerator.utils.reflection_utils import get_named_object, get_intersection_of_parameters

__author__ = 'mmb28'


def tokenize_data(data, tokenizer, corpus_ids):
    # param corpus_ids - list-like, names of the training corpus (and optional testing corpus), used for
    # retrieving pre-tokenized data from joblib cache
    x_tr, y_tr, x_test, y_test = data
    #todo this logic needs to be moved to feature extractor
    x_tr = tokenizer.tokenize_corpus(x_tr, corpus_ids[0])
    x_test = tokenizer.tokenize_corpus(x_test, corpus_ids[1])
    data = (x_tr, y_tr, x_test, y_test)
    return data


def load_text_data_into_memory(training_path, test_path=None, shuffle_targets=False):
    # read the raw text just once
    logging.info('Loading raw training set %s', training_path)
    x_train, y_train = _get_data_iterators(training_path, shuffle_targets=shuffle_targets)

    if test_path:
        logging.info('Loading raw test set %s' % test_path)
        x_test, y_test = _get_data_iterators(test_path, shuffle_targets=shuffle_targets)
    return (x_train, y_train, x_test, y_test), (training_path, test_path)


def load_tokenizer(normalise_entities=False, use_pos=True, coarse_pos=True, lemmatize=True,
                   lowercase=True, remove_stopwords=False, remove_short_words=False, joblib_caching=False):
    """
    Initialises the state of helper modules from a config object
    """

    if joblib_caching:
        memory = Memory(cachedir='.', verbose=0)
    else:
        memory = NoopTransformer()

    tok = tokenizers.XmlTokenizer(
        memory,
        normalise_entities=normalise_entities,
        use_pos=use_pos,
        coarse_pos=coarse_pos,
        lemmatize=lemmatize,
        lowercase=lowercase,
        remove_stopwords=remove_stopwords,
        remove_short_words=remove_short_words,
        use_cache=joblib_caching
    )
    return tok


def _get_data_iterators(path, shuffle_targets=False):
    """
    Returns iterators over the text of the data.

    :param path: The source folder to be read. Should contain data in the
     mallet format.
    :param shuffle_targets: If true, the true labels of the data set will be shuffled. This is useful as a
    sanity check
    """

    logging.info('Using a file content generator with source %(path)s' % locals())
    if not os.path.isdir(path):
        raise ValueError('The provided source path (%s) has to be a directory containing data in the mallet format'
                         ' (class per directory, document per file).')

    dataset = load_files(path, shuffle=False)
    logging.info('Targets are: %s', dataset.target_names)
    data_iterable = dataset.data
    if shuffle_targets:
        logging.warn('RANDOMIZING TARGETS')
        random.shuffle(dataset.target)
    targets_iterable = dataset.target

    return data_iterable, targets_iterable


def get_vector_source(conf, vector_source=None):
    vectors_exist_ = conf['feature_selection']['ensure_vectors_exist']
    handler_ = conf['feature_extraction']['decode_token_handler']
    if 'signified' in handler_.lower() or vectors_exist_:
        # vectors are needed either at decode time (signified handler) or during feature selection
        paths = conf['vector_sources']['unigram_paths']
        precomputed = conf['vector_sources']['precomputed']

        if not paths:
            raise ValueError('You must provide at least one neighbour source because you requested %s '
                             ' and ensure_vectors_exist=%s' % (handler_, vectors_exist_))
        if any('events' in x for x in paths) and precomputed:
            logging.warn('Possible configuration error: you requested precomputed '
                         'thesauri to be used but passed in the following files: \n%s', paths)

        if not precomputed:
            # load unigram vectors and initialise required composers based on these vectors
            if paths:
                logging.info('Loading unigram vector sources')
                unigram_source = UnigramVectorSource(paths,
                                                     reduce_dimensionality=conf['vector_sources'][
                                                         'reduce_dimensionality'],
                                                     dimensions=conf['vector_sources']['dimensions'])

            composers = []
            for section in conf['vector_sources']:
                if 'composer' in section and conf['vector_sources'][section]['run']:
                    # the object must only take keyword arguments
                    composer_class = get_named_object(section)
                    args = get_intersection_of_parameters(composer_class, conf['vector_sources'][section])
                    args['unigram_source'] = unigram_source
                    composers.append(composer_class(**args))
            if composers and not vector_source:
                # if a vector_source has not been predefined
                vector_source = CompositeVectorSource(
                    composers,
                    conf['vector_sources']['sim_threshold'],
                    conf['vector_sources']['include_self'],
                )
        else:
            filename = 'shelf%d' % hash(tuple(paths))
            if not os.path.exists(filename):
                logging.info('Loading precomputed neighbour sources')
                entry_types_to_load = conf['vector_sources']['entry_types_to_load']
                if not entry_types_to_load:
                    entry_types_to_load = ContainsEverything()

                vector_source = PrecomputedSimilaritiesVectorSource.from_file(
                    thesaurus_files=paths,
                    sim_threshold=conf['vector_sources']['sim_threshold'],
                    include_self=conf['vector_sources']['include_self'],
                    allow_lexical_overlap=conf['vector_sources']['allow_lexical_overlap'],
                    row_filter=lambda x, y: y.type in entry_types_to_load
                )
                vector_source.th.to_shelf(filename)
            vector_source = filename
    else:
        if not vector_source:
            # if a vector source has not been passed in and has not been initialised, then init it to avoid
            # accessing empty things
            vector_source = []
    return vector_source