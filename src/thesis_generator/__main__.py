#!/usr/bin/python
'''
Created on Oct 18, 2012

@author: ml249
'''
import os
import sys
import shutil
from glob import glob
import numpy as np
import logging
from logging import StreamHandler
from logging.handlers import RotatingFileHandler
import inspect

import validate
from configobj import ConfigObj

from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files

from joblib import Memory

from thesis_generator import config
from thesis_generator.utils import (get_named_object,
                                    LeaveNothingOut,
                                    ChainCallable)


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
                logger.debug('Retrieving input generator for name '
                             '\'%(input_gen)s\'' % locals())

                data_iterable = get_named_object(input_gen)(kwargs['source'])
                targets_iterable = np.asarray(
                    [t for t in data_iterable.targets()],
                    dtype=np.int)
                data_iterable = data_iterable.documents()
            except (ValueError, ImportError):
                logger.warn('No input generator found for name '
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

                dataset = load_files(source)
                data_iterable = dataset.data
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


def get_crossvalidation_iterator(config, x_vals, y_vals):
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
    logger.info('Building crossvalidation iterator')
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

    validation_indices = reduce(lambda l, (head, tail):
                                l + range(head, tail), validation_data, [])

    mask = np.zeros(y_vals.shape[0])  # we only mask the rows
    mask[validation_indices] = 1 # mask has 1 where the data point should be
    # used for validation and not for training/testing

    seen_data_mask = mask == 0
    dataset_size = np.sum(seen_data_mask)
    targets_seen = y_vals[seen_data_mask]
    if k < 0:
        logger.warn('crossvalidation.k not specified, defaulting to 1')
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
            logger.warn(
                'crossvalidation.ratio not specified,defaulting to 0.8')
            ratio = 0.8
        iterator = cross_validation.Bootstrap(dataset_size,
                                              n_iter=int(k),
                                              train_size=ratio)
    elif cv_type == 'oracle':
        iterator = LeaveNothingOut(dataset_size, dataset_size)
    else:
        raise ValueError(
            'Unrecognised crossvalidation type \'%(cv_type)s\'. The supported '
            'types are \'kfold\', \'skfold\', \'loo\', \'bootstrap\' and '
            '\'oracle\'')


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


def _build_vectorizer(call_args, feature_extraction_conf, pipeline_list):
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

    #    required_args = {}
    #    required_args.update({'%s' % arg: val
    #                          for arg, val in feature_extraction_conf.items()
    #                          if val != '' and arg in initialize_args})

    pipeline_list.append(('vect', vectorizer()))


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


def _build_pipeline(classifier_name, feature_extr_conf, feature_sel_conf,
                    dim_red_conf, classifier_conf):
    """
    Builds a pipeline consisting of
        - feature extractor
        - optional feature selection
        - optional dimensionality reduction
        - classifier
    """
    call_args = {}
    pipeline_list = []

    _build_vectorizer(call_args, feature_extr_conf,
                      pipeline_list)

    _build_feature_selector(call_args, feature_sel_conf,
                            pipeline_list)
    _build_dimensionality_reducer(call_args, dim_red_conf,
                                  pipeline_list)
    # include a classifier in the pipeline regardless of whether we are doing
    # feature selection/dim. red. or not
    clf = get_named_object(classifier_name)
    initialize_args = inspect.getargspec(clf.__init__)[0]
    call_args.update({'clf__%s' % arg: val
                      for arg, val in classifier_conf[classifier_name].items()
                      if val != '' and arg in initialize_args})
    pipeline_list.append(('clf', clf()))
    pipeline = Pipeline(pipeline_list)
    pipeline.set_params(**call_args)

#    logger.info('Preprocessing pipeline is %s', pipeline)
    return pipeline


def fit_pipeline(pipeline, x, y):
    pipeline.fit(x, y)


def run_tasks(args, configuration):
    """
    Runs all commands specified in the configuration file
    """

    # **********************************
    # CLEAN OUTPUT DIRECTORY
    # **********************************
    if args.clean and os.path.exists(args.output):
        logger.info('Cleaning output directory %s' % glob(args.output))
        shutil.rmtree(args.output)

    # **********************************
    # CREATE OUTPUT DIRECTORY
    # **********************************
    if not os.path.exists(args.output):
        logger.info('Creating output directory %s' % glob(args.output))
        os.makedirs(args.output)

        # TODO this needs to be redone after the scikits integration is
        # complete
    #    _write_config_file(args)

    # **********************************
    # ADD classpath TO SYSTEM PATH
    # **********************************
    for path in args.classpath.split(os.pathsep):
        logger.info('Adding (%s) to system path' % glob(path))
        sys.path.append(os.path.abspath(path))

    # get a reference to the joblib cache object
    mem_cache = Memory(cachedir=args.output, verbose=0)

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
        options = {}
        options['input'] = configuration['feature_extraction']['input']
        options['input_generator'] = configuration['feature_extraction'][
                                     'input_generator']
        options['source'] = args.source
        logger.info('Loading data set')
        x_vals, y_vals = cached_get_data_generators(**options)
        if args.test:
            logger.info('Loading test set')
            #  change where we read files from
            options['source'] = args.test
            x_test, y_test = cached_get_data_generators(**options)
        del options

    # **********************************
    # CROSSVALIDATION
    # **********************************
    # todo need to make sure that for several classifier the crossvalidation
    # iterator stays consistent across all classifiers

    # CREATE CROSSVALIDATION ITERATOR
    crossvalidate_cached = mem_cache.cache(get_crossvalidation_iterator)
    cv_iterator, validate_indices, x_vals_seen,\
    y_vals_seen = crossvalidate_cached(
        configuration['crossvalidation'], x_vals, y_vals)

    logger.debug("Starting training with %d documents" % len(x_vals_seen))

    scores = []
    for clf_name in configuration['classifiers']:
        if not configuration['classifiers'][clf_name]:
            continue

        #  ignore disabled classifiers
        if not configuration['classifiers'][clf_name]['run']:
            logger.warn('Ignoring classifier %s' % clf_name)
            continue

        logger.info('Classifier: %s' % clf_name)
        # DO FEATURE SELECTION/DIMENSIONALITY REDUCTION FOR CROSSVALIDATION
        # DATA
        pipeline = _build_pipeline(clf_name,
                                   configuration['feature_extraction'],
                                   configuration['feature_selection'],
                                   configuration['dimensionality_reduction'],
                                   configuration['classifiers'])

        if args.test:
            #  no crossvalidation, train on one set and test on the other
            logger.info('Evaluating on test set of size %s' % len(x_test))
            cached_fit = mem_cache.cache(fit_pipeline)
            cached_fit(pipeline, x_vals_seen, y_vals_seen)
            eval = ChainCallable(configuration['evaluation'])
            # Making a singleton tuple with the tuple of interest as the
            # only item
            for metric, score in eval(y_test, pipeline.predict(x_test))\
            .items():
                scores.append(
                    [clf_name.split('.')[-1], metric.split('.')[-1],
                     score])
        else:
            # pass the (feature selector + classifier) pipeline for evaluation
            cached_cross_val_score = mem_cache.cache(cross_val_score)
            scores_this_clf = cached_cross_val_score(pipeline, x_vals_seen,
                                                     y_vals_seen,
                                                     ChainCallable(
                                                         configuration[
                                                         'evaluation']),
                                                     cv=cv_iterator, n_jobs=10,
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
    logger.info('Classifier scores are %s' % scores)
    analyze(scores, args.output, configuration['name'])

    # todo create a mallet classifier wrapper in python that works with
    # the scikit crossvalidation stuff (has fit and predict and
    # predict_probas functions)


def analyze(scores, output_dir, name):
    """
    Stores a csv, xls and png representation of the data set. Requires pandas
    """

    logger.info("Saving results to disk")

    from pandas import DataFrame
    import matplotlib.pyplot as plt

    df = DataFrame(scores, columns=['classifier', 'metric', 'score'])

    # store csv for futher processing
    df.to_csv(os.path.join(output_dir, '%s.out.csv' % name))
    df.to_excel(os.path.join(output_dir, '%s.out.xls' % name))

    # plot results
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    df.boxplot(by=['classifier', 'metric'], ax=axes)
    fig.autofmt_xdate(rotation=45, bottom=0.5)
    plt.savefig(os.path.join(output_dir, '%s.out.png' % name), format='png',

                dpi=150)


def _config_logger(output_path=None):
    logger = logging.getLogger(__name__)

    fmt = logging.Formatter(fmt=('%(asctime)s\t%(module)s.%(funcName)s '
                                 '(line %(lineno)d)\t%(levelname)s : %('
                                 'message)s'),
                            datefmt='%d.%m.%Y %H:%M:%S')

    sh = StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(fmt)

    if output_path is not None:
        fh = RotatingFileHandler(os.path.join(output_path, 'log.txt'),
                                 maxBytes=int(2 * 10e8),
                                 backupCount=5)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)
    return logger

if __name__ == '__main__':
    args = config.arg_parser.parse_args()
    if args.log_path.startswith('./'):
        args.log_path = os.path.join(args.output, args.log_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    logger = _config_logger(args.log_path)

    logger.info(
        'Reading configuration file from \'%s\', conf spec from \'conf/'
        '.confrc\'' % (glob(args.configuration)[0]))

    conf_parser = ConfigObj(args.configuration, configspec='conf/.confrc')
    validator = validate.Validator()
    result = conf_parser.validate(validator)

    # todo add a more helpful guide to what exactly went wrong with the conf
    # object
    if not result:
        logger.warn('Invalid configuration')
        sys.exit(1)

    run_tasks(args, conf_parser)
