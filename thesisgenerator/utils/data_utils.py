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
from discoutils.tokens import Token
from discoutils.cmd_utils import run_and_log_output
from discoutils.misc import is_gzipped
import numpy as np
import json
import gzip
from networkx.readwrite.json_graph import node_link_data
from joblib import Parallel, delayed
from sklearn.datasets import load_files
from thesisgenerator.plugins.tokenizers import XmlTokenizer, GzippedJsonTokenizer
from thesisgenerator.utils.conf_file_utils import parse_config_file
from thesisgenerator.utils.misc import force_symlink
from thesisgenerator.composers.vectorstore import DummyThesaurus


def tokenize_data(data, tokenizer, corpus_ids):
    """
    :param data: list of strings, contents of documents to tokenize
    :param tokenizer:
    :param corpus_ids:  list-like, names of the training corpus (and optional testing corpus), used for
    retrieving pre-tokenized data from joblib cache
    """
    x_tr, y_tr, x_test, y_test = data
    # todo this logic needs to be moved to feature extractor
    x_tr = tokenizer.tokenize_corpus(x_tr, corpus_ids[0])
    if x_test is not None and y_test is not None and corpus_ids[1] is not None:
        x_test = tokenizer.tokenize_corpus(x_test, corpus_ids[1])
    return x_tr, y_tr, x_test, y_test


def load_text_data_into_memory(training_path, test_path=None, shuffle_targets=False):
    x_train, y_train = _get_data_iterators(training_path, shuffle_targets=shuffle_targets)

    if test_path:
        logging.info('Loading raw test set %s' % test_path)
        x_test, y_test = _get_data_iterators(test_path, shuffle_targets=shuffle_targets)
    else:
        x_test, y_test = None, None
    return (x_train, y_train, x_test, y_test), (training_path, test_path)


def get_tokenizer_settings_from_conf(conf):
    return {'normalise_entities': conf['feature_extraction']['normalise_entities'],
            'use_pos': conf['feature_extraction']['use_pos'],
            'coarse_pos': conf['feature_extraction']['coarse_pos'],
            'lemmatize': conf['feature_extraction']['lemmatize'],
            'lowercase': conf['tokenizer']['lowercase'],
            'remove_stopwords': conf['tokenizer']['remove_stopwords'],
            'remove_short_words': conf['tokenizer']['remove_short_words'],
            'remove_long_words': conf['tokenizer']['remove_long_words']}


def get_tokenizer_settings_from_conf_file(conf_file):
    conf, _ = parse_config_file(conf_file)
    return get_tokenizer_settings_from_conf(conf)


def get_tokenized_data(training_path, tokenizer_conf, shuffle_targets=False,
                       test_data='', gzip_json=True, *args, **kwargs):
    if gzip_json:
        tokenizer = GzippedJsonTokenizer(**tokenizer_conf)
        x_tr, y_tr = tokenizer.tokenize_corpus(training_path)
        if test_data:
            x_test, y_test = tokenizer.tokenize_corpus(test_data)
        else:
            x_test, y_test = None, None
        return x_tr, np.array(y_tr), x_test, np.array(y_test) if y_test else y_test
    else:
        tokenizer = XmlTokenizer(**tokenizer_conf)
        raw_data, data_ids = load_text_data_into_memory(training_path=training_path,
                                                        test_path=test_data,
                                                        shuffle_targets=shuffle_targets)
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

    dataset = load_files(path, shuffle=False, load_content=False)
    logging.info('Targets are: %s', dataset.target_names)
    data_iterable = dataset.filenames
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
            thesaurus = Vectors.from_tsv(path, gzipped=conf['gzip_resources'], **params)
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
    from discoutils.misc import Delayed

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


def gzip_single_thesaurus(conf_file):
    conf, _ = parse_config_file(conf_file)
    th = conf['vector_sources']['neighbours_file']

    if os.path.exists(th):
        # need force in case output file exists
        if is_gzipped(th):
            # file is already gzipped, just symlink
            logging.info('Symlinking %s', th)
            force_symlink(th, th + '.gz')
        else:
            # don't modify old file
            run_and_log_output('gzip --force --best -c {0} > {0}.gz'.format(th))
    else:
        logging.warning('Thesaurus does not exist: %s', th)


def gzip_all_thesauri(n_jobs):
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

    Parallel(n_jobs=n_jobs)(delayed(gzip_single_thesaurus)(conf_file) for conf_file in thesauri.values())


def jsonify_single_labelled_corpus(conf_file):
    """
    Tokenizes an entire XML corpus (sentence segmented and dependency parsed), incl test and train chunk,
    and writes its content to a single JSON gzip-ed file,
     one document per line. Each line is a JSON array, the first value of which is the label of
     the document, and the rest are JSON representation of the dependency parse trees of
     each sentence in the document. The resultant document can be loaded with a GzippedJsonTokenizer.

    :param conf_file: A conf file. If this is specified, the other params will not be used, but
     will be read from the conf file instead
    :param tokenizer_conf: a dict to configure the tokenizer
    :param train_set: path to the training set in XML format
    :param test_set: test set. If this is present, it will be written to the same file as the training set.
    """
    conf, _ = parse_config_file(conf_file)
    train_set = conf['training_data']
    test_set = conf['test_data']
    tokenizer_conf = get_tokenizer_settings_from_conf_file(conf_file)

    def _token_encode(t):
        if isinstance(t, Token):
            d = t.__dict__
            d.update({'__token__': True})
            return d
        raise TypeError

    def _write_corpus_to_json(x_tr, y_tr, outfile):
        for document, label in zip(x_tr, y_tr):
            all_data = [label]
            for sent_parse_tree in document:
                data = node_link_data(sent_parse_tree)
                all_data.append(data)
            outfile.write(bytes(json.dumps(all_data, default=_token_encode), 'UTF8'))
            outfile.write(bytes('\n', 'UTF8'))

    # always load the dataset from XML
    x_tr, y_tr, x_test, y_test = get_tokenized_data(train_set,
                                                    tokenizer_conf,
                                                    test_data=test_set, gzip_json=False)
    with gzip.open('%s.gz' % train_set, 'wb') as outfile:
        _write_corpus_to_json(x_tr, y_tr, outfile)
        logging.info('Writing %s to gzip json', train_set)
        if x_test:
            _write_corpus_to_json(x_test, y_test, outfile)


def get_all_corpora():
    """
    Returns a dict of {corpus_path: conf_file}, where conf file is one of
    the conf files that uses that corpus
    :rtype: dict
    """
    all_conf_files = glob('conf/exp*/exp*_base.conf')
    corpora = dict()
    for conf_file in all_conf_files:
        conf, _ = parse_config_file(conf_file)
        corpus = conf['training_data']
        corpora[corpus] = conf_file
    if not corpora:
        raise ValueError('Could not find any labelled corpora')
    return corpora


def jsonify_all_labelled_corpora(n_jobs):
    corpora = get_all_corpora()
    logging.info(corpora)
    Parallel(n_jobs=n_jobs)(delayed(jsonify_single_labelled_corpus)(conf_file) for conf_file in corpora.values())

