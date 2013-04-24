#!/usr/bin/python
'''
Created on Oct 18, 2012

@author: ml249
'''
import os
import sys
import shutil
from glob import glob
import logging
from logging import StreamHandler
import inspect
from time import sleep

import numpy as np
from numpy.ma import hstack
import validate
from configobj import ConfigObj
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from joblib import Memory

import config
from thesisgenerator.plugins import tokenizers, thesaurus_loader
from thesisgenerator.plugins.dumpers import FeatureVectorsCsvDumper
from thesisgenerator.utils import (get_named_object,
                                   LeaveNothingOut,
                                   ChainCallable,
                                   PredefinedIndicesIterator,
                                   SubsamplingPredefinedIndicesIterator,
                                   NoopTransformer,
                                   get_confrc)


# **********************************
# FEATURE EXTRACTION / PARSE
# **********************************

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
                logging.getLogger('root').debug(
                    'Retrieving input generator for name '
                    '\'%(input_gen)s\'' % locals())

                data_iterable = get_named_object(input_gen)(kwargs['source'])
                targets_iterable = np.asarray(
                    [t for t in data_iterable.targets()],
                    dtype=np.int)
                data_iterable = data_iterable.documents()
            except (ValueError, ImportError):
                logging.getLogger('root').warn(
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
                logging.getLogger('root').info(
                    'Targets are: %s' % dataset.target_names)
                data_iterable = dataset.data
                if kwargs['shuffle_targets']:
                    import random

                    logging.getLogger('root').warn('RANDOMIZING TARGETS')
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


def get_crossvalidation_iterator(config, x_vals, y_vals, x_test=None,
                                 y_test=None):
    """
    Returns a crossvalidation iterator, which contains a list of
    (train_indices, test_indices) that can be used to slice
    a dataset to perform crossvalidation. Additionally,
    returns the original data that was passed in and a mask specifying what
    data points should be used for validation.

    The method of splitting for CV is determined by what is specified in the
    conf file. The splitting of data in train/test/validate set is not done
    in this function- here we only return a mask for the validation data
    and an iterator for the train/test data.
    The full text is provided as a parameter so that joblib can cache the
    call to this function.
    """
    logging.getLogger('root').info('Building crossvalidation iterator')
    cv_type = config['type']
    k = config['k']

    if (config['validation_slices'] != '' and
                config['validation_slices'] is not None):
        # the data should be treated as a stream, which means that it should
        # not
        # be reordered and it should be split into a seen portion and an unseen
        # portion separated by a virtual 'now' point in the stream
        validation_data = get_named_object(config['validation_slices'])
        validation_data = validation_data(x_vals, y_vals)
    else:
        validation_data = [(0, 0)]

    validation_indices = reduce(lambda l, (head, tail): l + range(head, tail),
                                validation_data, [])

    if x_test is not None and y_test is not None:
        logging.getLogger('root').warn('You have requested test set to be '
                                       'used for evaluation.')
        if cv_type != 'test_set' and cv_type != 'subsampled_test_set':
            logging.getLogger('root').error('Wrong crossvalidation type. '
                                            'Only test_set or '
                                            'subsampled_test_set are '
                                            'permitted with a test set')
            sys.exit(1)

        train_indices = range(len(x_vals))
        test_indices = range(len(x_vals), len(x_vals) + len(x_test))
        x_vals.extend(x_test)
        y_vals = hstack([y_vals, y_test])

    mask = np.zeros(y_vals.shape[0])  # we only mask the rows
    mask[validation_indices] = 1 # mask has 1 where the data point should be
    # used for validation and not for training/testing

    seen_data_mask = mask == 0
    dataset_size = np.sum(seen_data_mask)
    targets_seen = y_vals[seen_data_mask]
    if k < 0:
        logging.getLogger('root').warn(
            'crossvalidation.k not specified, defaulting to 1')
        k = 1
    if cv_type == 'kfold':
        iterator = cross_validation.KFold(dataset_size, int(k))
    elif cv_type == 'skfold':
        iterator = cross_validation.StratifiedKFold(targets_seen, int(k))
    elif cv_type == 'loo':
        iterator = cross_validation.LeaveOneOut(dataset_size, int(k))
    elif cv_type == 'bootstrap':
        ratio = config['ratio']
        if k < 0:
            logging.getLogger('root').warn(
                'crossvalidation.ratio not specified,defaulting to 0.8')
            ratio = 0.8
        iterator = cross_validation.Bootstrap(dataset_size,
                                              n_iter=int(k),
                                              train_size=ratio)
    elif cv_type == 'oracle':
        iterator = LeaveNothingOut(dataset_size, dataset_size)
    elif cv_type == 'test_set' and x_test is not None and y_test is not None:
        iterator = PredefinedIndicesIterator(train_indices, test_indices)
    elif cv_type == 'subsampled_test_set' and x_test is not None and y_test is not None:
        iterator = SubsamplingPredefinedIndicesIterator(y_vals,
                                                        train_indices,
                                                        test_indices, int(k),
                                                        config['sample_size'],
                                                        config['random_state'])
    else:
        raise ValueError(
            'Unrecognised crossvalidation type \'%(cv_type)s\'. The supported '
            'types are \'kfold\', \'skfold\', \'loo\', \'bootstrap\', '
            '\'test_set\', \'subsampled_test_set\' and \'oracle\'')


    # Pick out the non-validation data from x_vals. This requires x_vals
    # to be cast to a format that supports slicing, such as the compressed
    # sparse row format (converting to that is also fast).
    seen_indices = range(targets_seen.shape[0])
    seen_indices = sorted(set(seen_indices) - set(validation_indices))
    x_vals = [x_vals[index] for index in seen_indices]
    # y_vals is a row vector, need to transpose it to get the same shape as
    # x_vals
    y_vals = y_vals[:, seen_indices].transpose()

    return iterator, validation_indices, x_vals, y_vals


def _build_vectorizer(id, call_args, feature_extraction_conf, pipeline_list,
                      output_dir, debug):
    """
    Builds a vectorized that converts raw text to feature vectors. The
    parameters for the vectorizer are specified in the *feature extraction*
    section of the configuration file. These will be matched to those of the
     *__init__* method of the    vectorizer class and the matching keywords
     are passed to the vectorizer. The non-matching arguments are simply
     ignored.

     The vectorizer converts a corpus into a term frequency matrix. A given
     source corpus is converted into a term frequency matrix and
     returned as a numpy *coo_matrix*.The value of the *vectorizer* field
        in the main configuration file is used as the transformer class.
        This class
        can be anything but has to implement the methods *fit*,
        *transform* and *fit_transform* as per scikit-learn.
    """
    vectorizer = get_named_object(feature_extraction_conf['vectorizer'])

    # todo preprocessor needs to be expanded into a callable name
    # todo analyzer needs to be expanded into a callable name
    # todo tokenizer needs to be expanded into a callable name
    # todo vocabulary needs to be a complex data type - this should be
    # allowed to be a file reference
    # todo dtype should be expanded into a numpy type

    # get the names of the arguments that the vectorizer class takes
    # todo the object must only take keyword arguments
    initialize_args = inspect.getargspec(vectorizer.__init__)[0]
    call_args.update({'vect__%s' % arg: val
                      for arg, val in feature_extraction_conf.items()
                      if val != '' and arg in initialize_args})

    pipeline_list.append(('vect', vectorizer()))
    call_args['vect__log_vocabulary'] = False

    # global postvect_dumper_added_already
    if debug:# and not postvect_dumper_added_already:
        logging.getLogger('root').info('Will perform post-vectorizer data dump')
        pipeline_list.append(
            ('dumper', FeatureVectorsCsvDumper(id, output_dir)))
        # postvect_dumper_added_already = True
        call_args['vect__log_vocabulary'] = True # tell the vectorizer it
        # needs to persist some information (used by the postvect dumper)
        # this is needed because object in the pipeline are isolated
        call_args['vect__pipe_id'] = id


def _build_feature_selector(call_args, feature_selection_conf, pipeline_list):
    """
    If feature selection is required, this function appends a selector
    object to pipeline_list and its configuration to configuration. Note this
     function modifies (appends to) its input arguments
    """
    if feature_selection_conf['run']:
        method = get_named_object(feature_selection_conf['method'])
        scoring_func = get_named_object(
            feature_selection_conf['scoring_function'])

        # the parameters for steps in the Pipeline are defined as
        # <component_name>__<arg_name> - the Pipeline (which is actually a
        # BaseEstimator) takes care of passing the correct arguments down
        # along the pipeline, provided there are no name clashes between the
        # keyword arguments of two consecutive transformers.

        initialize_args = inspect.getargspec(method.__init__)[0]
        call_args.update({'fs__%s' % arg: val
                          for arg, val in feature_selection_conf.items()
                          if val != '' and arg in initialize_args})

        pipeline_list.append(('fs', method(scoring_func)))


def _build_dimensionality_reducer(call_args, dimensionality_reduction_conf,
                                  pipeline_list):
    """
      If dimensionality reduciton is required, this function appends a reducer
      object to pipeline_list and its configuration to configuration. Note this
       function modifies (appends to) its input arguments
      """

    if dimensionality_reduction_conf['run']:
        dr_method = get_named_object(dimensionality_reduction_conf['method'])
        initialize_args = inspect.getargspec(dr_method.__init__)[0]
        call_args.update({'dr__%s' % arg: val
                          for arg, val in dimensionality_reduction_conf.items()
                          if val != '' and arg in initialize_args})
        pipeline_list.append(('dr', dr_method()))


def _build_pipeline(id, classifier_name, feature_extr_conf, feature_sel_conf,
                    dim_red_conf, classifier_conf, output_dir, debug):
    """
    Builds a pipeline consisting of
        - feature extractor
        - optional feature selection
        - optional dimensionality reduction
        - classifier
    """
    call_args = {}
    pipeline_list = []

    _build_vectorizer(id, call_args, feature_extr_conf,
                      pipeline_list, output_dir, debug)

    _build_feature_selector(call_args, feature_sel_conf,
                            pipeline_list)
    _build_dimensionality_reducer(call_args, dim_red_conf,
                                  pipeline_list)
    # include a classifier in the pipeline regardless of whether we are doing
    # feature selection/dim. red. or not
    if classifier_name:
        clf = get_named_object(classifier_name)
        initialize_args = inspect.getargspec(clf.__init__)[0]
        call_args.update({'clf__%s' % arg: val
                          for arg, val in
                          classifier_conf[classifier_name].items()
                          if val != '' and arg in initialize_args})
        pipeline_list.append(('clf', clf()))
    pipeline = Pipeline(pipeline_list)
    pipeline.set_params(**call_args)

    logging.getLogger('root').debug('Pipeline is:\n %s', pipeline)
    return pipeline


def run_tasks(configuration, data=None):
    """
    Runs all commands specified in the configuration file
    """
    logging.getLogger('root').info('running tasks')
    # get a reference to the joblib cache object, if caching is not disabled
    # else build a dummy object which just returns its arguments unchanged
    if configuration['joblib_caching']:
        mem_cache = Memory(cachedir=configuration['output_dir'], verbose=0)
    else:
        # op = type("JoblibDummy", (object,), {"cache": lambda self, x: x})
        mem_cache = NoopTransformer()

    # retrieve the actions that should be run by the framework
    actions = configuration.keys()

    # **********************************
    # LOADING RAW TEXT
    # **********************************
    if ('feature_extraction' in actions and
            configuration['feature_extraction']['run']):
        # todo should figure out which values to ignore,
        # currently use all (args + section_options)
        cached_get_data_generators = mem_cache.cache(_get_data_iterators)

        # create the keyword argument list the action should be run with, it is
        # very important that all relevant argument:value pairs are present
        # because joblib uses the hashed argument list to lookup cached results
        # of computations that have been executed previously
        if data:
            logging.getLogger('root').info('Using pre-loaded raw data set')
            x_vals, y_vals, x_test, y_test = data
        else:
            options = {}
            options['input'] = configuration['feature_extraction']['input']
            options['shuffle_targets'] = configuration['shuffle_targets']
            try:
                options['input_generator'] = \
                    configuration['feature_extraction']['input_generator']
            except KeyError:
                options['input_generator'] = ''
            options['source'] = configuration['training_data']

            logging.getLogger('root').info('Loading raw training set')
            x_vals, y_vals = cached_get_data_generators(**options)
            if configuration['test_data']:
                logging.getLogger('root').info('Loading raw test set')
                #  change where we read files from
                options['source'] = configuration['test_data']
                # ensure that only the training data targets are shuffled
                options['shuffle_targets'] = False
                x_test, y_test = cached_get_data_generators(**options)
            del options


    # **********************************
    # CROSSVALIDATION
    # **********************************
    # todo need to make sure that for several classifier the crossvalidation
    # iterator stays consistent across all classifiers

    # CREATE CROSSVALIDATION ITERATOR
    crossvalidate_cached = mem_cache.cache(get_crossvalidation_iterator)
    cv_iterator, validate_indices, x_vals_seen, y_vals_seen = \
        crossvalidate_cached(
            configuration['crossvalidation'], x_vals, y_vals, x_test, y_test)

    del x_vals
    del y_vals # only the non-validation data should be used from now on

    scores = []
    for i, clf_name in enumerate(configuration['classifiers']):
        if not configuration['classifiers'][clf_name]:
            continue

        #  ignore disabled classifiers
        if not configuration['classifiers'][clf_name]['run']:
            logging.getLogger('root').warn('Ignoring classifier %s' % clf_name)
            continue

        logging.getLogger('root').info(
            '------------------------------------------------------------')
        logging.getLogger('root').info('Building pipeline')
        pipeline = _build_pipeline(i, clf_name,
                                   configuration['feature_extraction'],
                                   configuration['feature_selection'],
                                   configuration['dimensionality_reduction'],
                                   configuration['classifiers'],
                                   configuration['output_dir'],
                                   configuration['debug'])

        # pass the (feature selector + classifier) pipeline for evaluation
        logging.getLogger('root').info(
            '***Fitting pipeline for %s' % clf_name)
        cached_cross_val_score = mem_cache.cache(cross_val_score)
        scores_this_clf = \
            cached_cross_val_score(pipeline, x_vals_seen, y_vals_seen,
                                   ChainCallable(
                                       configuration['evaluation']),
                                   cv=cv_iterator, n_jobs=1,
                                   verbose=0)

        for run_number in range(len(scores_this_clf)):
            a = scores_this_clf[run_number]
            # If there is just one metric specified in the conf file a is a
            # 0-D numpy array and needs to be indexed as [()]. Otherwise it
            # is a dict
            mydict = a[()] if hasattr(a, 'shape') and len(
                a.shape) < 1 else a
            for metric, score in mydict.items():
                scores.append(
                    [clf_name.split('.')[-1], metric.split('.')[-1],
                     score])
    logging.getLogger('root').info('Classifier scores are %s' % scores)
    return 0, analyze(scores, configuration['output_dir'],
                      configuration['name'])


def analyze(scores, output_dir, name):
    """
    Stores a csv and xls representation of the data set. Requires pandas
    """

    logging.getLogger('root').info(
        "Analysing results and saving to %s" % output_dir)

    from pandas import DataFrame

    cleaned_scores = []
    for result in scores:
        clf, metric, vals = result
        if len(result[2].shape) < 1:
            # the value is a scalar, let it be
            cleaned_scores.append(result)
        else:
            # the value is a list, e.g. per-class precision/recall/F1
            for id, val in enumerate(vals):
                # todo verify the correct class id is inserted here
                cleaned_scores.append([clf, '%s-class%d' % (metric, id), val])

    df = DataFrame(cleaned_scores, columns=['classifier', 'metric', 'score'])
    grouped = df.groupby(['classifier', 'metric'])
    from numpy import mean, std

    res = grouped['score'].aggregate({'score_mean': mean, 'score_std': std})

    # store csv for futher processing
    csv = os.path.join(output_dir, '%s.out.csv' % name)
    res.to_csv(csv, na_rep='-1')
    # df.to_excel(os.path.join(output_dir, '%s.out.xls' % name))

    return csv


def _config_logger(output_path=None, name='log'):
    newly_created_logger = logging.getLogger('root')

    # for parallelisation purposes we need to remove all the handlers that were
    # possibly added to the root logger OF THE PROCESS on previous runs,
    # otherwise logs will get mangled between different runs
    # print newly_created_logger.handlers
    # for some reason removing them once doesn't seem to be enough sometimes
    for i in range(5):
        for handler in newly_created_logger.handlers:
            newly_created_logger.removeHandler(handler)
        sleep(0.1)
    assert len(newly_created_logger.handlers) == 0

    fmt = logging.Formatter(fmt=('%(asctime)s\t%(module)s.%(funcName)s '
                                 '(line %(lineno)d)\t%(levelname)s : %('
                                 'message)s'),
                            datefmt='%d.%m.%Y %H:%M:%S')

    sh = StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    newly_created_logger.addHandler(sh)

    class MyFilter(object):
        """
        A logging filter which accepts messages with a level *LOWER* than the
         one specified at construction time
        """

        def __init__(self, level):
            self.__level = level

        def filter(self, logRecord):
            return logRecord.levelno <= self.__level

    if output_path is not None:
        log_file = os.path.join(output_path, '%s.log' % name)
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        #        fh.addFilter(MyFilter(logging.DEBUG))
        #   fh1 = logging.FileHandler(os.path.join(output_path, 'log-info.txt'),
        #                                  mode='w')
        #        fh1.setLevel(logging.INFO)
        #        fh1.setFormatter(fmt)
        newly_created_logger.addHandler(fh)

    newly_created_logger.setLevel(logging.DEBUG)
    return newly_created_logger
    # else:
    #     return log


def _prepare_output_directory(clean, output):
    # **********************************
    # CLEAN OUTPUT DIRECTORY
    # **********************************
    if clean and os.path.exists(output):
        logging.getLogger('root').info(
            'Cleaning output directory %s' % glob(output))
        shutil.rmtree(output)

    # **********************************
    # CREATE OUTPUT DIRECTORY
    # **********************************
    if not os.path.exists(output):
        logging.getLogger('root').info(
            'Creating output directory %s' % glob(output))
        os.makedirs(output)


def _prepare_classpath(classpath):
    # **********************************
    # ADD classpath TO SYSTEM PATH
    # **********************************
    for path in classpath.split(os.pathsep):
        logging.getLogger('root').info(
            'Adding (%s) to system path' % glob(path))
        sys.path.append(os.path.abspath(path))


def parse_config_file(conf_file):
    configspec_file = get_confrc(conf_file)
    config = ConfigObj(conf_file, configspec=configspec_file)
    validator = validate.Validator()
    result = config.validate(validator)
    # todo add a more helpful guide to what exactly went wrong with the conf
    # object
    if not result:
        print 'Invalid configuration'
        sys.exit(1)
    return config, configspec_file


def _init_utilities_state(config):
    """
    Initialises the state of helper modules from a config object
    """
    tokenizers.normalise_entities = config['feature_extraction'][
        'normalise_entities']
    tokenizers.use_pos = config['feature_extraction']['use_pos']
    tokenizers.coarse_pos = config['feature_extraction']['coarse_pos']
    tokenizers.lemmatize = config['feature_extraction']['lemmatize']
    tokenizers.lowercase = config['tokenizer']['lowercase']

    thesaurus_loader.thesaurus_files = config['feature_extraction'][
        'thesaurus_files']
    # thesaurus_loader.use_pos = config['feature_extraction']['use_pos']
    # thesaurus_loader.coarse_pos = config['feature_extraction']['coarse_pos']
    thesaurus_loader.sim_threshold = config['feature_extraction'][
        'sim_threshold']
    thesaurus_loader.k = config['feature_extraction']['k']
    thesaurus_loader.include_self = config['feature_extraction']['include_self']


def go(conf_file, log_dir, data=None, classpath='', clean=False):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    config, configspec_file = parse_config_file(conf_file)

    log = _config_logger(log_dir, config['name'])
    log.info(
        'Reading configuration file from \'%s\', conf spec from \'%s\''
        % (glob(conf_file)[0], configspec_file))
    _init_utilities_state(config)
    output = config['output_dir']
    _prepare_output_directory(clean, output)
    _prepare_classpath(classpath)
    status, msg = run_tasks(config, data)
    shutil.copy(conf_file, output)
    return status, msg


postvect_dumper_added_already = False

if __name__ == '__main__':
    args = config.arg_parser.parse_args()
    log_dir = args.log_path
    conf_file = args.configuration
    classpath = args.classpath
    clean = args.clean

    go(conf_file, log_dir, classpath=classpath, clean=clean)



    # runs a full pipeline in-process for profiling purposes
#    from thesis_generator.plugins.bov import ThesaurusVectorizer
#    options = {}
#    options['input'] = conf_parser['feature_extraction']['input']
#    options['input_generator'] = conf_parser['feature_extraction'][
#                                 'input_generator']
#    options['source'] = args.source
#    log.info('Loading raw training set')
#    x_vals, y_vals = _get_data_iterators(**options)
#    if args.test:
#        log.info('Loading raw test set')
#        #  change where we read files from
#        options['source'] = args.test
#        x_test, y_test = _get_data_iterators(**options)
#    del options
#
#    v = ThesaurusVectorizer(conf_parser['feature_extraction']['thesaurus_files'])
#    t = v.load_thesauri()
#    v.fit(x_vals)
