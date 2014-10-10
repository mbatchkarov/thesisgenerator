import argparse
from collections import ChainMap
from glob import glob
from hashlib import md5
import logging
import os
import random
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
from discoutils.thesaurus_loader import Vectors
from discoutils.misc import Delayed
import numpy as np
from joblib import Memory, Parallel, delayed
from sklearn.datasets import load_files
from thesisgenerator.classifiers import NoopTransformer
from thesisgenerator.plugins import tokenizers
from thesisgenerator.utils.conf_file_utils import parse_config_file
from thesisgenerator.composers.vectorstore import DummyThesaurus


def tokenize_data(data, tokenizer, corpus_ids):
    logging.info('Tokenizer starting')
    # param corpus_ids - list-like, names of the training corpus (and optional testing corpus), used for
    # retrieving pre-tokenized data from joblib cache
    x_tr, y_tr, x_test, y_test = data
    # todo this logic needs to be moved to feature extractor
    x_tr = tokenizer.tokenize_corpus(x_tr, corpus_ids[0])
    if x_test is not None and y_test is not None and corpus_ids[1] is not None:
        x_test = tokenizer.tokenize_corpus(x_test, corpus_ids[1])
    logging.info('Tokenizer finished')
    return x_tr, y_tr, x_test, y_test


def load_text_data_into_memory(training_path, test_path=None, shuffle_targets=False):
    # read the raw text just once
    logging.info('Loading raw training set %s', training_path)
    x_train, y_train = _get_data_iterators(training_path, shuffle_targets=shuffle_targets)

    if test_path:
        logging.info('Loading raw test set %s' % test_path)
        x_test, y_test = _get_data_iterators(test_path, shuffle_targets=shuffle_targets)
    else:
        x_test, y_test = None, None
    return (x_train, y_train, x_test, y_test), (training_path, test_path)


def load_tokenizer(normalise_entities=False, use_pos=True, coarse_pos=True, lemmatize=True,
                   lowercase=True, remove_stopwords=False, remove_short_words=False,
                   remove_long_words=False, joblib_caching=False):
    """
    Initialises the state of helper modules from a config object
    """

    if joblib_caching:
        memory = Memory(cachedir='.', verbose=1)
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
        remove_long_words=remove_long_words,
        use_cache=joblib_caching
    )
    return tok


def get_tokenized_data(training_data, normalise_entities,
                       use_pos, coarse_pos, lemmatize, lowercase, remove_stopwords,
                       remove_short_words, remove_long_words, shuffle_targets,
                       test_data=None, *args, **kwargs):
    raw_data, data_ids = load_text_data_into_memory(training_path=training_data, test_path=test_data,
                                                    shuffle_targets=shuffle_targets)
    tokenizer = load_tokenizer(joblib_caching=True,  # remove this parameter
                               normalise_entities=normalise_entities,
                               use_pos=use_pos,
                               coarse_pos=coarse_pos,
                               lemmatize=lemmatize,
                               lowercase=lowercase,
                               remove_stopwords=remove_stopwords,
                               remove_short_words=remove_short_words,
                               remove_long_words=remove_long_words
    )
    return tokenize_data(raw_data, tokenizer, data_ids)


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
        raise ValueError('The provided source path %s has to be a directory containing data in the mallet format'
                         ' (class per directory, document per file).' % path)

    dataset = load_files(path, shuffle=False)
    logging.info('Targets are: %s', dataset.target_names)
    data_iterable = dataset.data
    if shuffle_targets:
        logging.warning('RANDOMIZING TARGETS')
        random.shuffle(dataset.target)

    return data_iterable, np.array(dataset.target_names)[dataset.target]


def get_thesaurus(conf):
    vectors_exist_ = conf['feature_selection']['must_be_in_thesaurus']
    handler_ = conf['feature_extraction']['decode_token_handler']
    random_thes = conf['feature_extraction']['random_neighbour_thesaurus']
    path = conf['vector_sources']['neighbours_file']
    use_shelf = conf['vector_sources']['use_shelf']

    thesaurus = None
    if random_thes:
        return DummyThesaurus(k=conf['feature_extraction']['k'], constant=False)

    if 'signified' in handler_.lower() or vectors_exist_:
        # vectors are needed either at decode time (signified handler) or during feature selection

        if not path and not random_thes:
            raise ValueError('You must provide at least one neighbour source because you requested %s '
                             ' and must_be_in_thesaurus=%s' % (handler_, vectors_exist_))

        params = conf['vector_sources']
        # delays the loading from disk/de-shelving until the resource is needed. The Delayed object also makes it
        # possible to get either Vectors or Thesaurus into the pipeline, and there is no need to pass any parameters
        # that relate to IO further down the pipeline
        if use_shelf:
            thesaurus = load_and_shelve_thesaurus(path, **params)
        else:
            # single we are running single-threaded, might as well read this in now
            # returning a delayed() will cause the file to be read for each CV fold
            # thesaurus = Delayed(Vectors, Vectors.from_tsv, path, **params)
            thesaurus = Vectors.from_tsv(path, **params)
    if not thesaurus:
        # if a vector source has not been passed in and has not been initialised, then init it to avoid
        # accessing empty things
        logging.warning('RETURNING AN EMPTY THESAURUS')
        thesaurus = []
    return thesaurus


def load_and_shelve_thesaurus(path, **kwargs):
    """
    Parses and then shelves a thesaurus file. Reading from it is much faster and memory efficient than
    keeping it in memory. Returns a callable that returns the thesaurus
    :rtype: Delayed
    """
    # built-in hash has randomisation enabled by default on py>=3.3
    filename = 'shelf_%s' % md5(path.encode('utf8')).hexdigest()
    # shelve may add an extension or split the file in bits with different extensions
    search_paths = glob('%s*' % filename)
    if search_paths:  # there are files that match that name
        logging.info('Returning pre-shelved object %s for %s', filename, path)
    else:
        # that shelf does not exist, create it
        th = Vectors.from_tsv(path, **kwargs)
        logging.info('Shelving %s to %s', path, filename)
        if len(th) > 0:  # don't bother with empty thesauri
            th.to_shelf(filename)
    return Delayed(Vectors, Vectors.from_shelf_readonly, filename, **kwargs)


def shelve_single_thesaurus(conf_file):
    conf, _ = parse_config_file(conf_file)
    th = conf['vector_sources']['neighbours_file']

    if os.path.exists(th):
        load_and_shelve_thesaurus(th, **conf['vector_sources'])
    else:
        logging.warning('Thesaurus does not exist: %s', th)


def shelve_all_thesauri(n_jobs):
    """
    Loads, parses and shelves all thesauri used in experiments.
    """
    # make sure thesauri that are used in multiple experiments are only shelved once
    all_conf_files = glob('conf/exp*/exp*_base.conf')
    thesauri = dict()
    for conf_file in all_conf_files:
        conf, _ = parse_config_file(conf_file)
        thes_file = conf['vector_sources']['neighbours_file']
        thesauri[thes_file] = conf_file

    Parallel(n_jobs=n_jobs)(delayed(shelve_single_thesaurus)(conf_file) for conf_file in thesauri.values())


def cache_single_labelled_corpus(conf_file, memory=None):
    if not memory:
        memory = Memory(cachedir='.', verbose=0)
    conf, _ = parse_config_file(conf_file)
    get_cached_tokenized_data = memory.cache(get_tokenized_data, ignore=['*', '**'])
    _ = get_cached_tokenized_data(**ChainMap(conf,
                                             conf['feature_extraction'],
                                             conf['tokenizer'],
                                             conf['feature_extraction']))


def cache_all_labelled_corpora(n_jobs):
    all_conf_files = glob('conf/exp*/exp*_base.conf')
    corpora = dict()
    for conf_file in all_conf_files:
        conf, _ = parse_config_file(conf_file)
        corpus = conf['training_data']
        corpora[corpus] = conf_file
    print(corpora)
    Parallel(n_jobs=n_jobs)(delayed(cache_single_labelled_corpus)(conf_file) for conf_file in corpora.values())

