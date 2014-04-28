#!/usr/bin/python
# coding=utf-8
"""
Created on Oct 18, 2012

@author: ml249
"""

# if one tries to run this script from the main project directory the
# thesisgenerator package would not be on the path, add it and try again
import pickle
import sys

from joblib import Parallel, delayed

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

from discoutils.misc import Bunch
from thesisgenerator.composers.feature_selectors import MetadataStripper
from thesisgenerator.utils.reflection_utils import get_named_object, get_intersection_of_parameters
from thesisgenerator.utils.misc import ChainCallable
from thesisgenerator.classifiers import LeaveNothingOut, PredefinedIndicesIterator, \
    SubsamplingPredefinedIndicesIterator, PicklingPipeline
from thesisgenerator.utils.conf_file_utils import set_in_conf_file, parse_config_file
from thesisgenerator.utils.data_utils import tokenize_data, load_text_data_into_memory, \
    load_tokenizer, get_vector_source
from thesisgenerator import config
from thesisgenerator.plugins.dumpers import FeatureVectorsCsvDumper


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

    dataset_size = len(x_vals)
    if k < 0:
        logging.warn('crossvalidation.k not specified, defaulting to 1')
        k = 1
    if cv_type == 'kfold':
        iterator = cross_validation.KFold(dataset_size, int(k))
    elif cv_type == 'skfold':
        iterator = cross_validation.StratifiedKFold(y_vals, int(k))
    elif cv_type == 'loo':
        iterator = cross_validation.LeaveOneOut(dataset_size, int(k))
    elif cv_type == 'bootstrap':
        ratio = config['ratio']
        if k < 0:
            logging.warn('crossvalidation.ratio not specified,defaulting to 0.8')
            ratio = 0.8
        iterator = cross_validation.Bootstrap(dataset_size, n_iter=int(k), train_size=ratio)
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

    return iterator, x_vals, y_vals


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

    # get the names of the arguments that the vectorizer class takes
    # the object must only take keyword arguments
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
      If dimensionality reduction is required, this function appends a reducer
      object to pipeline_list and its configuration to configuration. Note this
       function modifies (appends to) its input arguments
      """
    #  todo this isn't really needed and should probably be removed

    if dimensionality_reduction_conf['run']:
        dr_method = get_named_object(dimensionality_reduction_conf['method'])
        call_args.update(get_intersection_of_parameters(dr_method, dimensionality_reduction_conf, 'dr'))
        pipeline_list.append(('dr', dr_method()))


def _build_pipeline(cv_i, vector_source, feature_extr_conf, feature_sel_conf,
                    dim_red_conf, output_dir, debug,
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

    _build_vectorizer(cv_i, vector_source, init_args, fit_args, feature_extr_conf,
                      pipeline_list, output_dir, exp_name=exp_name)

    _build_feature_selector(vector_source, init_args, fit_args, feature_sel_conf, pipeline_list)
    _build_dimensionality_reducer(init_args, dim_red_conf, pipeline_list)

    # put the optional dumper after feature selection/dim. reduction
    if debug:
        logging.info('Will perform post-vectorizer data dump')
        pipeline_list.append(('dumper', FeatureVectorsCsvDumper(exp_name, cv_i, output_dir)))
        init_args['vect__pipe_id'] = cv_i

    # vectorizer will return a matrix (as usual) and some metadata for use with feature dumper/selector,
    # strip them before we proceed to the classifier
    pipeline_list.append(('stripper', MetadataStripper()))
    if vector_source:
        fit_args['stripper__vector_source'] = vector_source

    if feature_extr_conf['record_stats']:
        fit_args['vect__stats_hdf_file'] = 'stats-%s-cv%d' % (exp_name, cv_i)

    pipeline = PicklingPipeline(pipeline_list, exp_name) if debug else Pipeline(pipeline_list)
    pipeline.set_params(**init_args)

    logging.debug('Pipeline is:\n %s', pipeline)
    return pipeline, fit_args


def _build_classifiers(classifiers_conf):
    for i, clf_name in enumerate(classifiers_conf):
        if not classifiers_conf[clf_name]:
            continue
            #  ignore disabled classifiers
        if not classifiers_conf[clf_name]['run']:
            logging.debug('Ignoring classifier %s' % clf_name)
            continue
        clf = get_named_object(clf_name)
        init_args = get_intersection_of_parameters(clf, classifiers_conf[clf_name])
        yield clf(**init_args)


def _cv_loop(configuration, cv_i, score_func, test_idx, train_idx, vector_source, x_vals_seen, y_vals_seen):
    scores_this_cv_run = []
    pipeline, fit_params = _build_pipeline(cv_i, vector_source,
                                           configuration['feature_extraction'],
                                           configuration['feature_selection'],
                                           configuration['dimensionality_reduction'],
                                           configuration['output_dir'],
                                           configuration['debug'],
                                           exp_name=configuration['name'])
    # code below is a simplified version of sklearn's _cross_val_score
    X = x_vals_seen
    y = y_vals_seen
    X_train = [X[idx] for idx in train_idx]
    X_test = [X[idx] for idx in test_idx]
    # vectorize all data in advance, it's the same accross all classifiers
    matrix = pipeline.fit_transform(X_train, y[train_idx], **fit_params)
    test_matrix = pipeline.transform(X_test)
    stats = pipeline.named_steps['vect'].stats

    for clf in _build_classifiers(configuration['classifiers']):
        logging.info('Starting training of %s', clf)
        clf = clf.fit(matrix, y[train_idx])
        scores = score_func(y[test_idx], clf.predict(test_matrix))

        if configuration['feature_extraction']['record_stats']:
            inv_voc = {index: feature for (feature, index) in pipeline.named_steps['vect'].vocabulary_.items()}
            with open('%s.%s.pkl' % (stats.prefix, clf.__class__.__name__.split('.')[-1]), 'w') as outf:
                logging.info('Pickling trained classifier to %s', outf.name)
                b = Bunch(clf=clf, inv_voc=inv_voc)
                pickle.dump(b, outf)

        for metric, score in scores.items():
            scores_this_cv_run.append(
                [type(clf).__name__,
                 cv_i,
                 metric.split('.')[-1],
                 score])
        logging.info('Done with %s', clf)
    return scores_this_cv_run


def _analyze(scores, output_dir, name):
    """
    Stores a csv representation of the data set. Requires pandas
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

    if output_path is not None:
        log_file = os.path.join(output_path, '%s.log' % name)
        fh = logging.FileHandler(log_file, mode='w')
        if debug:
            fh.setLevel(logging.DEBUG)
        else:
            fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        newly_created_logger.addHandler(fh)

    newly_created_logger.setLevel(logging.DEBUG)
    return newly_created_logger


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

    log = _config_logger(log_dir, name=config['name'], debug=config['debug'])
    log.info('Reading configuration file from \'%s\', conf spec from \'%s\''
             % (glob(conf_file)[0], configspec_file))
    output = config['output_dir']
    _prepare_output_directory(clean, output)
    _prepare_classpath(classpath)
    shutil.copy(conf_file, output)

    # Runs all commands specified in the configuration file
    logging.info('Running tasks')

    # **********************************
    # LOADING RAW TEXT
    # **********************************
    x_tr, y_tr, x_test, y_test = data

    # CREATE CROSSVALIDATION ITERATOR
    cv_iterator, x_vals_seen, y_vals_seen = \
        _build_crossvalidation_iterator(config['crossvalidation'], x_tr, y_tr, x_test, y_test)
    all_scores = []
    score_func = ChainCallable(config['evaluation'])

    scores_over_cv = Parallel(n_jobs=n_jobs)(
        delayed(_cv_loop)(config, i, score_func, test_idx, train_idx, vector_source, x_vals_seen, y_vals_seen)
        for i, (train_idx, test_idx) in enumerate(cv_iterator)
    )
    all_scores.extend([score for one_set_of_scores in scores_over_cv for score in one_set_of_scores])
    output_file = _analyze(all_scores, config['output_dir'], config['name'])
    return output_file


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