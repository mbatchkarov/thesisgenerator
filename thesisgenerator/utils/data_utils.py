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
from discoutils.cmd_utils import run_and_log_output
from discoutils.misc import is_gzipped
import numpy as np
import pandas as pd
import json
import gzip
from joblib import Parallel, delayed
from sklearn.datasets import load_files
from thesisgenerator.plugins.tokenizers import XmlTokenizer, GzippedJsonTokenizer
from thesisgenerator.utils.conf_file_utils import parse_config_file
from thesisgenerator.utils.misc import force_symlink
from thesisgenerator.composers.vectorstore import RandomThesaurus
from thesisgenerator.plugins.bov import ThesaurusVectorizer
from thesisgenerator.plugins.multivectors import MultiVectors


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
                       test_data='', *args, **kwargs):
    """
    Loads data from either XML or compressed JSON
    :param gzip_json: set to True of False to force this method to read XML/JSON. Otherwise the type of
     input data is determined by the presence or absence of a .gz extension on the training_path
    :param args:
    """
    if is_gzipped(training_path):
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


def get_pipeline_fit_args(conf):
    """
    Builds a dict of resources that document vectorizers require at fit time. These currently include
    various kinds of distributional information, e.g. word vectors or cluster ID for words and phrases.
    Example:
    {'vector_source': <DenseVectors object>} or {'clusters': <pd.DataFrame of word clusters>}
    :param conf: configuration dict
    :raise ValueError: if the conf is wrong in any way
    """
    result = dict()
    vectors_exist = conf['feature_selection']['must_be_in_thesaurus']
    handler_ = conf['feature_extraction']['decode_token_handler']
    random_thes = conf['feature_extraction']['random_neighbour_thesaurus']
    vs_params = conf['vector_sources']
    vectors_path = vs_params['neighbours_file']
    clusters_path = vs_params['clusters_file']

    if 'Base' in handler_:
        # don't need vectors, this is a non-distributional experiment
        return result
    if vectors_path and clusters_path:
        raise ValueError('Cannot use both word vectors and word clusters')

    if random_thes:
        result['vector_source'] = RandomThesaurus(k=conf['feature_extraction']['k'])
    else:
        if vectors_path and clusters_path:
            raise ValueError('Cannot use both word vectors and word clusters')
        if 'signified' in handler_.lower() or vectors_exist:
            # vectors are needed either at decode time (signified handler) or during feature selection
            if not (vectors_path or clusters_path):
                raise ValueError('You must provide at least one source of distributional information '
                                 'because you requested %s and must_be_in_thesaurus=%s' % (handler_, vectors_exist))

    if len(vectors_path) == 1:
        # set up a row filter, if needed
        entries = vs_params['entries_of']
        if entries:
            entries = get_thesaurus_entries(entries)
            vs_params['row_filter'] = lambda x, y: x in entries
        result['vector_source'] = Vectors.from_tsv(vectors_path[0], **vs_params)
    if len(vectors_path) > 1:
        all_vect = [Vectors.from_tsv(p, **vs_params) for p in vectors_path]
        result['vector_source'] = MultiVectors(all_vect)

    if clusters_path:
        result['clusters'] = pd.read_hdf(clusters_path, key='clusters')

    return result


def get_thesaurus_entries(tsv_file):
    """
    Returns the set of entries contained in a thesaurus
    :param tsv_file: path to vectors file
    """
    return set(Vectors.from_tsv(tsv_file).keys())


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


def gzip_single_thesaurus(vectors_path):
    if os.path.exists(vectors_path):
        # need force in case output file exists
        if is_gzipped(vectors_path):
            # file is already gzipped, just symlink
            logging.info('Symlinking %s', vectors_path)
            force_symlink(vectors_path, vectors_path + '.gz')
        else:
            # don't modify old file
            run_and_log_output('gzip --force --best -c {0} > {0}.gz'.format(vectors_path))
    else:
        logging.warning('Thesaurus does not exist: %s', vectors_path)


def gzip_all_thesauri(n_jobs):
    """
    Loads, parses and shelves all thesauri used in experiments.
    """
    # make sure thesauri that are used in multiple experiments are only shelved once
    vector_paths = set(v.path for v in Vectors.select() if v.path)
    Parallel(n_jobs=n_jobs)(delayed(gzip_single_thesaurus)(conf_file) for conf_file in vector_paths)


def jsonify_single_labelled_corpus(corpus_path, tokenizer_conf=None):
    """
    Tokenizes an entire XML corpus (sentence segmented and dependency parsed), incl test and train chunk,
    and writes its content to a single JSON gzip-ed file,
    one document per line. Each line is a JSON array, the first value of which is the label of
    the document, and the rest are JSON representation of a list of lists, containing all document
    features of interest, e.g. nouns, adj, NPs, VPs, wtc.
    The resultant document can be loaded with a GzippedJsonTokenizer.

    :param corpus_path: path to the corpus
    """

    def _write_corpus_to_json(x_tr, y_tr, outfile):
        vect = ThesaurusVectorizer(min_df=1,
                                   train_time_opts={'extract_unigram_features': set('JNV'),
                                                    'extract_phrase_features': set(['AN', 'NN', 'VO', 'SVO'])})
        vect.extract_unigram_features = vect.train_time_opts['extract_unigram_features']
        vect.extract_phrase_features = vect.train_time_opts['extract_phrase_features']
        all_features = []
        for doc in x_tr:
            all_features.append([str(f) for f in vect.extract_features_from_token_list(doc)])

        for document, label in zip(all_features, y_tr):
            outfile.write(bytes(json.dumps([label, document]), 'UTF8'))
            outfile.write(bytes('\n', 'UTF8'))

    # load the dataset from XML
    if tokenizer_conf is None:
        tokenizer_conf = get_tokenizer_settings_from_conf_file('conf/exp1-superbase.conf')
    x_tr, y_tr, x_test, y_test = get_tokenized_data(corpus_path, tokenizer_conf)
    with gzip.open('%s.gz' % corpus_path, 'wb') as outfile:
        _write_corpus_to_json(x_tr, y_tr, outfile)
        logging.info('Writing %s to gzip json', corpus_path)
        if x_test:
            _write_corpus_to_json(x_test, y_test, outfile)


def get_all_corpora():
    """
    Returns a manually compiled list of all corpora used in experiments
    :rtype: list
    """
    return [
        '/lustre/scratch/inf/mmb28/thesisgenerator/sample-data/techtc100-clean/Exp_47456_497201-tagged',
        '/lustre/scratch/inf/mmb28/thesisgenerator/sample-data/techtc100-clean/Exp_324745_85489-tagged',
        '/lustre/scratch/inf/mmb28/thesisgenerator/sample-data/techtc100-clean/Exp_69753_85489-tagged',
        '/lustre/scratch/inf/mmb28/thesisgenerator/sample-data/techtc100-clean/Exp_186330_94142-tagged',
        '/lustre/scratch/inf/mmb28/thesisgenerator/sample-data/techtc100-clean/Exp_22294_25575-tagged',

        '/lustre/scratch/inf/mmb28/thesisgenerator/sample-data/reuters21578/r8-tagged-grouped',
        '/lustre/scratch/inf/mmb28/thesisgenerator/sample-data/movie-reviews-tagged',
        '/lustre/scratch/inf/mmb28/thesisgenerator/sample-data/amazon_grouped-tagged',
        '/lustre/scratch/inf/mmb28/thesisgenerator/sample-data/aclImdb-tagged',
    ]


def jsonify_all_labelled_corpora(n_jobs):
    corpora = get_all_corpora()
    logging.info(corpora)
    Parallel(n_jobs=n_jobs)(delayed(jsonify_single_labelled_corpus)(corpus) for corpus in corpora)
