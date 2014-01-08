#!/usr/bin/python
# coding=utf-8
"""
Created on Oct 18, 2012

@author: ml249
"""

# if one tries to run this script from the main project directory the
# thesisgenerator package would not be on the path, add it and try again
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

import os
import shutil
from glob import glob
import logging
from logging import StreamHandler
from time import sleep

from pandas import DataFrame
import numpy as np
from numpy.ma import hstack
from sklearn import cross_validation
from sklearn.pipeline import Pipeline

from thesisgenerator.composers.feature_selectors import MetadataStripper
from thesisgenerator.utils.reflection_utils import get_named_object, get_intersection_of_parameters
from thesisgenerator.utils.misc import ChainCallable
from thesisgenerator.classifiers import LeaveNothingOut, PredefinedIndicesIterator, SubsamplingPredefinedIndicesIterator, PicklingPipeline
from thesisgenerator.utils.conf_file_utils import set_in_conf_file, parse_config_file
from thesisgenerator.utils.data_utils import tokenize_data, load_text_data_into_memory, \
    load_tokenizer, get_vector_source
from thesisgenerator import config
from thesisgenerator.plugins.dumpers import FeatureVectorsCsvDumper
from thesisgenerator.plugins.crossvalidation import naming_cross_val_score


def _build_crossvalidation_iterator(config, x_vals, y_vals, x_test=None,
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
    logging.info('Building crossvalidation iterator')
    cv_type = config['type']
    k = config['k']

    validation_data = [(0, 0)]

    validation_indices = reduce(lambda l, (head, tail): l + range(head, tail),
                                validation_data, [])

    if x_test is not None and y_test is not None:
        logging.warn('You have requested test set to be used for evaluation.')
        if cv_type != 'test_set' and cv_type != 'subsampled_test_set':
            logging.error('Wrong crossvalidation type. Only test_set '
                          'or subsampled_test_set are permitted with a test set')
            sys.exit(1)

        x_vals = list(x_vals)
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
        logging.warn(
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
            logging.warn(
                'crossvalidation.ratio not specified,defaulting to 0.8')
            ratio = 0.8
        iterator = cross_validation.Bootstrap(dataset_size,
                                              n_iter=int(k),
                                              train_size=ratio)
    elif cv_type == 'oracle':
        iterator = LeaveNothingOut(dataset_size)
    elif cv_type == 'test_set' and x_test is not None and y_test is not None:
        iterator = PredefinedIndicesIterator(train_indices, test_indices)
    elif cv_type == 'subsampled_test_set' and \
                    x_test is not None and y_test is not None:
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
    y_vals = y_vals[seen_indices].transpose()

    return iterator, validation_indices, x_vals, y_vals


def _build_vectorizer(id, vector_source, init_args, fit_args, feature_extraction_conf, pipeline_list,
                      output_dir, exp_name=''):
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
    init_args.update(get_intersection_of_parameters(vectorizer, feature_extraction_conf, 'vect'))
    init_args['vect__exp_name'] = exp_name
    if vector_source:
        fit_args['vect__vector_source'] = vector_source

    pipeline_list.append(('vect', vectorizer()))


def _build_feature_selector(vector_source, init_args, fit_args, feature_selection_conf, pipeline_list):
    """
    If feature selection is required, this function appends a selector
    object to pipeline_list and its configuration to configuration. Note this
     function modifies (appends to) its input arguments
    """
    if feature_selection_conf['run']:
        method = get_named_object(feature_selection_conf['method'])
        scoring = feature_selection_conf.get('scoring_function')
        logging.info('Scoring function is %s', scoring)
        scoring_func = get_named_object(scoring) if scoring else None

        # the parameters for steps in the Pipeline are defined as
        # <component_name>__<arg_name> - the Pipeline (which is actually a
        # BaseEstimator) takes care of passing the correct arguments down
        # along the pipeline, provided there are no name clashes between the
        # keyword arguments of two consecutive transformers.

        init_args.update(get_intersection_of_parameters(method, feature_selection_conf, 'fs'))
        if vector_source:
            fit_args['fs__vector_source'] = vector_source
        logging.info('FS method is %s', method)
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
        call_args.update(get_intersection_of_parameters(dr_method, dimensionality_reduction_conf, 'dr'))
        pipeline_list.append(('dr', dr_method()))


def _build_pipeline(id, vector_source, classifier_name, feature_extr_conf, feature_sel_conf,
                    dim_red_conf, classifier_conf, output_dir, debug,
                    exp_name=''):
    """
    Builds a pipeline consisting of
        - feature extractor
        - optional feature selection
        - optional dimensionality reduction
        - classifier
    """
    init_args, fit_args = {}, {}
    pipeline_list = []

    _build_vectorizer(id, vector_source, init_args, fit_args, feature_extr_conf,
                      pipeline_list, output_dir, exp_name=exp_name)

    _build_feature_selector(vector_source, init_args, fit_args, feature_sel_conf, pipeline_list)
    _build_dimensionality_reducer(init_args, dim_red_conf, pipeline_list)

    # put the optional dumper after feature selection/dim. reduction
    if debug:
        logging.info('Will perform post-vectorizer data dump')
        pipeline_list.append(('dumper', FeatureVectorsCsvDumper(exp_name, id, output_dir)))
        init_args['vect__pipe_id'] = id

    # vectorizer will return a matrix (as usual) and some metadata for use with feature dumper/selector,
    # strip them before we proceed to the classifier
    pipeline_list.append(('stripper', MetadataStripper()))
    if vector_source:
        fit_args['stripper__vector_source'] = vector_source

    # include a classifier in the pipeline regardless of whether we are doing
    # feature selection/dim. red. or not
    if classifier_name:
        clf = get_named_object(classifier_name)
        init_args.update(get_intersection_of_parameters(clf, classifier_conf[classifier_name], 'clf'))
        pipeline_list.append(('clf', clf()))
    pipeline = PicklingPipeline(pipeline_list, exp_name) if debug else Pipeline(pipeline_list)
    pipeline.set_params(**init_args)

    logging.debug('Pipeline is:\n %s', pipeline)
    return pipeline, fit_args


def _run_tasks(configuration, n_jobs, data, vector_source):
    """
    Runs all commands specified in the configuration file
    """
    logging.info('running tasks')

    # retrieve the actions that should be run by the framework
    #actions = configuration.keys()

    # **********************************
    # LOADING RAW TEXT
    # **********************************
    x_tr, y_tr, x_test, y_test = data

    # CROSSVALIDATION
    # **********************************
    scores = []
    for i, clf_name in enumerate(configuration['classifiers']):
        if not configuration['classifiers'][clf_name]:
            continue

        #  ignore disabled classifiers
        if not configuration['classifiers'][clf_name]['run']:
            logging.warn('Ignoring classifier %s' % clf_name)
            continue

        logging.info('--------------------------------------------------------')
        logging.info('Building pipeline')

        # CREATE CROSSVALIDATION ITERATOR
        cv_iterator, validate_indices, x_vals_seen, y_vals_seen = \
            _build_crossvalidation_iterator(configuration['crossvalidation'],
                                            x_tr, y_tr, x_test,
                                            y_test)

        logging.info('Assigning id %d to classifier %s', i, clf_name)
        pipeline, fit_params = _build_pipeline(i, vector_source, clf_name,
                                               configuration['feature_extraction'],
                                               configuration['feature_selection'],
                                               configuration['dimensionality_reduction'],
                                               configuration['classifiers'],
                                               configuration['output_dir'],
                                               configuration['debug'],
                                               exp_name=configuration['name'])

        # pass the (feature selector + classifier) pipeline for evaluation
        logging.info('***Fitting pipeline for %s' % clf_name)

        # the pipeline is cloned for each fold of the CV iterator, but its fit arguments aren't
        # I've added a clone call for the fit args to the CV function, so that the fit arguments are not
        # going to be shared between pipeline folds, but are shared between all estimators inside
        # a pipeline during a fold
        logging.debug('Identity of vector source is %d', id(vector_source))
        if vector_source:
            try:
                logging.debug('The BallTree is %s', vector_source.nbrs)
            except AttributeError:
                logging.debug('The vector source is %s', vector_source)
                # pass the same vector source to the vectorizer, feature selector and metadata stripper
                # that way the stripper can call vector_source.populate() after the feature selector has had its say,
                # and that update source will then be available to the vectorizer at decode time
                #fit_params = {
                #    'vect__vector_source': vector_source,
                #    'fs__vector_source': vector_source,
            #    'stripper__vector_source': vector_source,
        #}
        scores_this_clf = naming_cross_val_score(
            pipeline, x_vals_seen,
            y_vals_seen,
            ChainCallable(configuration['evaluation']),
            cv=cv_iterator, n_jobs=n_jobs,
            verbose=0, fit_params=fit_params) # pass resource here

        for run_number, a in scores_this_clf:
            # If there is just one metric specified in the conf file a is a
            # 0-D numpy array and needs to be indexed as [()]. Otherwise it
            # is a dict
            mydict = a[()] if hasattr(a, 'shape') and len(a.shape) < 1 else a
            for metric, score in mydict.items():
                scores.append(
                    [clf_name.split('.')[-1],
                     run_number,
                     metric.split('.')[-1],
                     score])
        del pipeline
        del scores_this_clf
    logging.info('Classifier scores are %s', scores)
    return 0, _analyze(scores, configuration['output_dir'], configuration['name'])


def _analyze(scores, output_dir, name):
    """
    Stores a csv and xls representation of the data set. Requires pandas
    """

    logging.info("Analysing results and saving to %s", output_dir)
    cleaned_scores = []
    for result in scores:
        clf, run_no, metric, vals = result
        if np.isscalar(vals):
            # the value is a scalar, let it be
            cleaned_scores.append(result)
        else:
            # the value is a list, e.g. per-class precision/recall/F1
            for id, val in enumerate(vals):
                # todo verify the correct class id is inserted here
                cleaned_scores.append([clf,
                                       run_no,
                                       '%s-class%d' % (metric, id),
                                       val])

    # save raw results
    df = DataFrame(cleaned_scores,
                   columns=['classifier', 'cv_no', 'metric', 'score'])
    csv = os.path.join(output_dir, '%s.out-raw.csv' % name)
    df.to_csv(csv, na_rep='-1')

    # now calculate mean and std
    grouped = df.groupby(['classifier', 'metric'])
    from numpy import mean, std

    res = grouped['score'].aggregate({'score_mean': mean, 'score_std': std})

    # store csv for futher processing
    csv = os.path.join(output_dir, '%s.out.csv' % name)
    res.to_csv(csv, na_rep='-1')
    # df.to_excel(os.path.join(output_dir, '%s.out.xls' % name))

    return csv


def _config_logger(output_path=None, name='log', debug=False):
    newly_created_logger = logging.getLogger()

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
        if debug:
            fh.setLevel(logging.DEBUG)
        else:
            fh.setLevel(logging.INFO)
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
        logging.info('Cleaning output directory %s', glob(output))
        shutil.rmtree(output)

    # **********************************
    # CREATE OUTPUT DIRECTORY
    # **********************************
    if not os.path.exists(output):
        logging.info('Creating output directory %s', glob(output))
        os.makedirs(output)


def _prepare_classpath(classpath):
    # **********************************
    # ADD classpath TO SYSTEM PATH
    # **********************************
    for path in classpath.split(os.pathsep):
        logging.info('Adding (%s) to system path' % glob(path))
        sys.path.append(os.path.abspath(path))


def go(conf_file, log_dir, data, vector_source, classpath='', clean=False, n_jobs=1):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    config, configspec_file = parse_config_file(conf_file)

    #if config['debug'] and config['crossvalidation']['run'] and \
    #                config['crossvalidation']['k'] > 1:
    #    raise ValueError('Cannot crossvalidate and debug at the same time')
    # because all folds run at the same time and write to the same debug
    # file

    log = _config_logger(log_dir, name=config['name'], debug=config['debug'])
    log.info('Reading configuration file from \'%s\', conf spec from \'%s\''
             % (glob(conf_file)[0], configspec_file))
    output = config['output_dir']
    _prepare_output_directory(clean, output)
    _prepare_classpath(classpath)
    status, msg = _run_tasks(config, n_jobs, data, vector_source)
    shutil.copy(conf_file, output)
    return status, msg


postvect_dumper_added_already = False

if __name__ == '__main__':
    # for debugging single sub-experiments only

    args = config.arg_parser.parse_args()
    log_dir = args.log_path
    conf_file = args.configuration
    classpath = args.classpath
    clean = args.clean

    # set debug=True, disable crossvalidation and enable coverage recording
    set_in_conf_file(conf_file, 'debug', True)
    set_in_conf_file(conf_file, ['crossvalidation', 'k'], 1)
    set_in_conf_file(conf_file, ['feature_extraction', 'record_stats'], True)

    # only leave one classifier enabled to speed things up
    set_in_conf_file(conf_file, ['classifiers', 'sklearn.naive_bayes.MultinomialNB', 'run'], True)
    set_in_conf_file(conf_file, ['classifiers', 'sklearn.svm.LinearSVC', 'run'], False)
    set_in_conf_file(conf_file, ['classifiers', 'sklearn.naive_bayes.BernoulliNB', 'run'], False)
    set_in_conf_file(conf_file, ['classifiers', 'thesisgenerator.classifiers.MultinomialNBWithBinaryFeatures', 'run'],
                     False)
    set_in_conf_file(conf_file, ['classifiers', 'sklearn.linear_model.LogisticRegression', 'run'], False)
    set_in_conf_file(conf_file, ['classifiers', 'sklearn.neighbors.KNeighborsClassifier', 'run'], False)

    conf, configspec_file = parse_config_file(conf_file)

    data, data_id = load_text_data_into_memory(conf['training_data'], conf['test_data'])
    tokenizer = load_tokenizer(
        joblib_caching=conf['joblib_caching'],
        normalise_entities=conf['feature_extraction']['normalise_entities'],
        use_pos=conf['feature_extraction']['use_pos'],
        coarse_pos=conf['feature_extraction']['coarse_pos'],
        lemmatize=conf['feature_extraction']['lemmatize'],
        lowercase=conf['tokenizer']['lowercase'],
        remove_stopwords=conf['tokenizer']['remove_stopwords'],
        remove_short_words=conf['tokenizer']['remove_short_words'])
    data = tokenize_data(data, tokenizer, data_id)
    vector_store = get_vector_source(conf)
    go(conf_file, log_dir, data, vector_store, classpath=classpath, clean=clean, n_jobs=1)