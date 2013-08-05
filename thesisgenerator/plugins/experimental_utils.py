# coding=utf-8


# if one tries to run this script from the main project directory the
# thesisgenerator package would not be on the path, add it and try again
import logging
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from thesisgenerator.plugins.file_generators import _exp16_file_iterator, _exp1_file_iterator, _vary_training_size_file_iterator
from joblib import Memory
import numpy as np
from sklearn.datasets import load_files
from thesisgenerator.plugins import thesaurus_loader, tokenizers

import glob
from itertools import chain
import os
from joblib import Parallel, delayed

from numpy import nonzero

from thesisgenerator.__main__ import go, parse_config_file
from thesisgenerator.utils import get_susx_mysql_conn, get_named_object, NoopTransformer
from thesisgenerator.plugins.dumpers import *
from thesisgenerator.plugins.consolidator import consolidate_results


def _init_utilities_state(config):
    """
    Initialises the state of helper modules from a config object
    """

    if config['joblib_caching']:
        memory = Memory(cachedir='.', verbose=0)
    else:
        memory = NoopTransformer()

    th1 = thesaurus_loader.Thesaurus(
        config['feature_extraction']['train_thesaurus_files'],
        sim_threshold=config['feature_extraction']['sim_threshold'],
        include_self=config['feature_extraction']['include_self'])

    tok = tokenizers.XmlTokenizer(
        memory,
        normalise_entities=config['feature_extraction']['normalise_entities'],
        use_pos=config['feature_extraction']['use_pos'],
        coarse_pos=config['feature_extraction']['coarse_pos'],
        lemmatize=config['feature_extraction']['lemmatize'],
        thesaurus=th1,
        lowercase=config['tokenizer']['lowercase'],
        keep_only_IT=config['tokenizer']['keep_only_IT'],
        remove_stopwords=config['tokenizer']['remove_stopwords'],
        remove_short_words=config['tokenizer']['remove_short_words'],
        use_cache=config['joblib_caching']
    )
    return th1, tok


def inspect_thesaurus_effect(outdir, clf_name, thesaurus_file, pipeline,
                             predicted, x_test):
    """
    Evaluates the performance of a classifier with and without the thesaurus
    that backs its vectorizer
    """

    # remove the thesaurus
    pipeline.named_steps['vect'].thesaurus = {}
    predicted2 = pipeline.predict(x_test)

    with open('%s/before_after_%s.csv' %
                      (outdir, thesaurus_file), 'a') as outfile:
        outfile.write('DocID,')
        outfile.write(','.join([str(x) for x in range(len(predicted))]))
        outfile.write('\n')
        outfile.write('%s+Thesaurus,' % clf_name)
        outfile.write(','.join([str(x) for x in predicted.tolist()]))
        outfile.write('\n')
        outfile.write('%s-Thesaurus,' % clf_name)
        outfile.write(','.join([str(x) for x in predicted2.tolist()]))
        outfile.write('\n')
        outfile.write('Decisions changed: %d' % (
            nonzero(predicted - predicted2)[0].shape[0]))
        outfile.write('\n')


def load_text_data_into_memory(options):
    logging.info('Loading raw training set')
    x_train, y_train = _get_data_iterators(**options)
    if options['test_data']:
        logging.info('Loading raw test set')
        #  change where we read files from
        options['source'] = options['test_data']
        # ensure that only the training data targets are shuffled
        options['shuffle_targets'] = False
        x_test, y_test = _get_data_iterators(**options)
    return x_train, y_train, x_test, y_test


def _get_data_iterators(**kwargs):
    """
    Returns iterators over the text of the data.

    The *input_generator* option in the main configuration file is an
    optional argument for *feature_extract*. It should specify
    the fully qualified name of a generator class with two
    methods *documents* and *classes*. If the vectorizer's
    *input* value 'content' the *input_generator* will be used to
    feed the raw documents to the vectorizer.

    If the *input_generator* is not defined and the *input* field is
    *content* the source folder specified on the command line will be used
    as the input.  The source folder should in this case contain data in the
     mallet format. The same applies if the value of *input* is *filename*.

    See the documentation of the CountVectorizer in
    sklearn.feature_extraction.text for details on the parameter values.
    """

    def _filename_generator(file_list):
        for f in file_list:
            yield f

    def _content_generator(file_list):
        for f in file_list:
            with open(f, 'rb') as fh:
                yield fh.read()

    if kwargs['input'] == 'content' or kwargs['input'] == '':
        try:
            input_gen = kwargs['input_generator']
            source = kwargs['source']
            try:
                logging.debug(
                    'Retrieving input generator for name '
                    '\'%(input_gen)s\'' % locals())

                data_iterable = get_named_object(input_gen)(kwargs['source'])
                targets_iterable = np.asarray(
                    [t for t in data_iterable.targets()],
                    dtype=np.int)
                data_iterable = data_iterable.documents()
            except (ValueError, ImportError):
                logging.warn(
                    'No input generator found for name '
                    '\'%(input_gen)s\'. Using a file content '
                    'generator with source \'%(source)s\'' % locals())
                if not os.path.isdir(source):
                    raise ValueError('The provided source path (%s) has to be '
                                     'a directory containing data in the '
                                     'mallet '
                                     'format (class per directory, document '
                                     'per file). If you intended to load the '
                                     'contents of the file (%s) instead '
                                     'change '
                                     'the input type in main.conf to '
                                     '\'content\'')

                dataset = load_files(source, shuffle=False)
                logging.info('Targets are: %s' % dataset.target_names)
                data_iterable = dataset.data
                if kwargs['shuffle_targets']:
                    import random

                    logging.warn('RANDOMIZING TARGETS')
                    random.shuffle(dataset.target)
                targets_iterable = dataset.target
        except KeyError:
            raise ValueError('Can not find a name for an input generator. '
                             'When the input type for feature extraction is '
                             'defined as content, an input_generator must '
                             'also '
                             'be defined. The defined input_generator should '
                             'produce raw documents.')
    elif kwargs['input'] == 'filename':
        raise NotImplementedError("The order of data and targets is wrong, "
                                  "do not use this keyword")
    elif kwargs['input'] == 'file':
        raise NotImplementedError(
            'The input type \'file\' is not supported yet.')
    else:
        raise NotImplementedError(
            'The input type \'%s\' is not supported yet.' % kwargs['input'])

    return data_iterable, targets_iterable


def tokenize_data(base_conf_file):
    config_obj, configspec_file = parse_config_file(base_conf_file)
    # read the raw text just once
    options = {'input': config_obj['feature_extraction']['input'],
               'shuffle_targets': config_obj['shuffle_targets']}
    try:
        options['input_generator'] = config_obj['feature_extraction'][
            'input_generator']
    except KeyError:
        options['input_generator'] = ''
    options['source'] = config_obj['training_data']
    if config_obj['test_data']:
        options['test_data'] = config_obj['test_data']
    print 'Loading training data...'
    x_tr, y_tr, x_test, y_test = load_text_data_into_memory(options)
    thesaurus, tokenizer = _init_utilities_state(config_obj)
    x_tr = map(tokenizer.tokenize, x_tr)
    x_test = map(tokenizer.tokenize, x_test)
    data = (x_tr, y_tr, x_test, y_test)
    return data, thesaurus


def _clear_old_files(i, prefix):
    """
    clear old conf, logs and result files for this experiment
    """
    for f in glob.glob('%s/conf/exp%d/exp%d_base-variants/*' % (prefix, i, i)):
        os.remove(f)
    for f in glob.glob('%s/conf/exp%d/output/*' % (prefix, i)):
        os.remove(f)
    for f in glob.glob('%s/conf/exp%d/logs/*' % (prefix, i)):
        os.remove(f)


def run_experiment(i, num_workers=4,
                   predefined_sized=[],
                   prefix='/Volumes/LocalDataHD/mmb28/NetBeansProjects/thesisgenerator'):
    print 'RUNNING EXPERIMENT %d' % i
    # on local machine

    exp1_thes_pattern = '%s/../Byblo-2.1.0/exp6-1*/*sims.neighbours.strings' % \
                        prefix
    # on cluster
    import platform

    hostname = platform.node()
    if 'apollo' in hostname or 'node' in hostname:
        # set the paths automatically when on the cluster
        prefix = '/mnt/lustre/scratch/inf/mmb28/thesisgenerator'
        exp1_thes_pattern = '%s/../FeatureExtrationToolkit/exp6-1*/*sims.' \
                            'neighbours.strings' % prefix
        num_workers = 30

    sizes = chain(range(2, 11, 2), range(20, 101, 10))
    if i == 0:
        # exp0 is for debugging only, we don't have to do much
        sizes = [10, 20]#range(10, 31, 10)
        num_workers = 1
    if predefined_sized:
        sizes = predefined_sized

    # ----------- EXPERIMENT 1 -----------
    base_conf_file = '%s/conf/exp%d/exp%d_base.conf' % (prefix, i, i)
    if i == 1:
        conf_file_iterator = _exp1_file_iterator(exp1_thes_pattern,
                                                 '%s/conf/exp1/exp1_base.conf' % prefix)

    # ----------- EXPERIMENTS 2-14 -----------
    elif i == 0 or 1 < i <= 14 or 17 <= i <= 28:
        conf_file_iterator = _vary_training_size_file_iterator(sizes, i, base_conf_file)
    elif i == 16:
        conf_file_iterator = _exp16_file_iterator(base_conf_file)
    else:
        raise ValueError('No such experiment number: %d' % i)

    _clear_old_files(i, prefix)
    data, thesurus = tokenize_data(base_conf_file)
    # run the data through the pipeline
    Parallel(n_jobs=num_workers)(delayed(go)(new_conf_file, log_file, data, thesurus) for
                                 new_conf_file, log_file in conf_file_iterator)

    # ----------- CONSOLIDATION -----------
    output_dir = '%s/conf/exp%d/output/' % (prefix, i)
    csv_out_fh = open(os.path.join(output_dir, "summary%d.csv" % i), "w")

    if not ('apollo' in hostname or 'node' in hostname):
        output_db_conn = get_susx_mysql_conn()
        writer = ConsolidatedResultsSqlAndCsvWriter(i, csv_out_fh,
                                                    output_db_conn)
    else:
        writer = ConsolidatedResultsCsvWriter(csv_out_fh)
    consolidate_results(
        writer,
        '%s/conf/exp%d/exp%d_base-variants' % (prefix, i, i),
        '%s/conf/exp%d/logs/' % (prefix, i),
        output_dir
    )


if __name__ == '__main__':
    run_experiment(int(sys.argv[1]))

