# !/usr/bin/python
# coding=utf-8
"""
Created on Oct 18, 2012

@author: ml249
"""

# if one tries to run this script from the main project directory the
# thesisgenerator package would not be on the path, add it and try again
import dbm
import pickle
import sys

from joblib import Parallel, delayed
from datetime import datetime

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

from discoutils.misc import Bunch, Delayed
from thesisgenerator.composers.feature_selectors import MetadataStripper
from thesisgenerator.utils.reflection_utils import get_named_object, get_intersection_of_parameters
from thesisgenerator.utils.misc import ChainCallable
from thesisgenerator.classifiers import (LeaveNothingOut, PredefinedIndicesIterator,
                                         SubsamplingPredefinedIndicesIterator, PicklingPipeline)
from thesisgenerator.utils.conf_file_utils import set_in_conf_file, parse_config_file
from thesisgenerator.utils.data_utils import (tokenize_data, load_text_data_into_memory,
                                              load_tokenizer, get_thesaurus)
from thesisgenerator.utils.misc import update_dict_according_to_mask
from thesisgenerator import config
from thesisgenerator.plugins.dumpers import FeatureVectorsCsvDumper


def _build_crossvalidation_iterator(config, y_train, y_test=None):
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
    # todo document parameters and check all caller comply to the intended usage pattern
    logging.info('Building crossvalidation iterator')
    cv_type = config['type']
    k = config['k']
    dataset_size = len(y_train)

    if y_test is not None:
        logging.warning('You have requested test set to be used for evaluation.')
        if cv_type != 'test_set' and cv_type != 'subsampled_test_set':
            logging.error('Wrong crossvalidation type. Only test_set '
                          'or subsampled_test_set are permitted with a test set')
            sys.exit(1)

        train_indices = range(dataset_size)
        test_indices = range(dataset_size, dataset_size + len(y_test))
        y_train = hstack([y_train, y_test])
        dataset_size += len(y_test)

    random_state = config['random_state']
    if k < 0:
        logging.warning('crossvalidation.k not specified, defaulting to 1')
        k = 1
    if cv_type == 'kfold':
        iterator = cross_validation.KFold(dataset_size, n_folds=int(k), random_state=random_state)
    elif cv_type == 'skfold':
        iterator = cross_validation.StratifiedKFold(y_train, n_folds=int(k), random_state=random_state)
    elif cv_type == 'loo':
        iterator = cross_validation.LeaveOneOut(dataset_size, int(k))
    elif cv_type == 'bootstrap':
        ratio = config['ratio']
        if k < 0:
            logging.warning('crossvalidation.ratio not specified,defaulting to 0.8')
            ratio = 0.8
        iterator = cross_validation.Bootstrap(dataset_size, n_iter=int(k),
                                              train_size=ratio, random_state=random_state)
    elif cv_type == 'oracle':
        iterator = LeaveNothingOut(dataset_size)
    elif cv_type == 'test_set' and y_test is not None:
        iterator = PredefinedIndicesIterator(train_indices, test_indices)
    elif cv_type == 'subsampled_test_set' and y_test is not None:
        iterator = SubsamplingPredefinedIndicesIterator(y_train,
                                                        train_indices,
                                                        test_indices, int(k),
                                                        config['sample_size'],
                                                        config['random_state'])
    else:
        raise ValueError(
            'Unrecognised crossvalidation type \'%(cv_type)s\'. The supported '
            'types are \'kfold\', \'skfold\', \'loo\', \'bootstrap\', '
            '\'test_set\', \'subsampled_test_set\' and \'oracle\'')

    return iterator, y_train


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


def _build_pipeline(cv_i, vector_source, feature_extr_conf, feature_sel_conf, output_dir, debug, exp_name=''):
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
        fit_args['vect__stats_hdf_file'] = 'statistics/stats-%s-cv%d' % (exp_name, cv_i)

    pipeline = PicklingPipeline(pipeline_list, exp_name) if debug else Pipeline(pipeline_list)
    pipeline.set_params(**init_args)

    logging.debug('Pipeline is:\n %s', pipeline)
    return pipeline, fit_args


def _build_classifiers(classifiers_conf):
    for i, clf_name in enumerate(classifiers_conf):
        if not classifiers_conf[clf_name]:
            continue
            # ignore disabled classifiers
        if not classifiers_conf[clf_name]['run']:
            logging.debug('Ignoring classifier %s' % clf_name)
            continue
        clf = get_named_object(clf_name)
        init_args = get_intersection_of_parameters(clf, classifiers_conf[clf_name])
        yield clf(**init_args)


def _cv_loop(log_dir, config, cv_i, score_func, test_idx, train_idx, vector_source:Delayed, X, y):
    _config_logger(log_dir,
                   name='{}-cv{}'.format(config['name'], cv_i),
                   debug=config['debug'])

    if isinstance(vector_source, Delayed):
        # build the actual object now (in the worker sub-process)
        vector_source = vector_source()

    scores_this_cv_run = []
    pipeline, fit_params = _build_pipeline(cv_i, vector_source,
                                           config['feature_extraction'],
                                           config['feature_selection'],
                                           config['output_dir'],
                                           config['debug'],
                                           exp_name=config['name'])
    # code below is a simplified version of sklearn's _cross_val_score
    train_text = [X[idx] for idx in train_idx]
    test_text = [X[idx] for idx in test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # vectorize all data in advance, it's the same across all classifiers
    tr_matrix = pipeline.fit_transform(train_text, y_train, **fit_params)
    # Update the vocabulary of the vectorizer. The feature selector may remove some vocabulary entries,
    # but the vectorizer will be unaware of this. Because the vectorizer's logic is conditional on whether
    # features are IV or OOV, this is a problem. (The issue is caused by the fact that the vectorizer and feature
    # selector are not independent. The special logic of the vectorizer should have been implemented as a third
    # transformer, but this would require too much work at this time.
    if 'fs' in pipeline.named_steps:
        pipeline.named_steps['vect'].vocabulary_ = pipeline.named_steps['fs'].vocabulary_
    test_matrix = pipeline.transform(test_text)
    stats = pipeline.named_steps['vect'].stats


    # remove documents with too few features
    to_keep_train = tr_matrix.A.sum(axis=1) >= config['min_train_features']
    logging.info('%d/%d train documents have enough features', sum(to_keep_train), len(y_train))
    tr_matrix = tr_matrix[to_keep_train, :]
    y_train = y_train[to_keep_train]

    # the slice above may remove all occurences of a feature,
    # e.g. when it only occurs in one document (very common) and the document
    # doesn't have enough features. Drop empty columns in the term-doc matrix
    column_mask = tr_matrix.A.sum(axis=0) > 0
    tr_matrix = tr_matrix[:, column_mask]

    voc = update_dict_according_to_mask(pipeline.named_steps['vect'].vocabulary_, column_mask)
    inv_voc = {index: feature for (feature, index) in voc.items()}

    # do the same for the test set
    to_keep_test = test_matrix.A.sum(axis=1) >= config['min_test_features']  # todo need unit test
    logging.info('%d/%d test documents have enough features', sum(to_keep_test), len(y_test))
    test_matrix = test_matrix[to_keep_test, :]
    y_test = y_test[to_keep_test]

    for clf in _build_classifiers(config['classifiers']):
        logging.info('Starting training of %s', clf)
        clf = clf.fit(tr_matrix, y_train)
        predictions = clf.predict(test_matrix)
        scores = score_func(y_test, predictions)

        tr_set_scores = score_func(y_train, clf.predict(tr_matrix))
        logging.info('Training set scores: %r', tr_set_scores)

        if config['feature_extraction']['record_stats']:
            # if a feature selectors exist, use its vocabulary
            # step_name = 'fs' if 'fs' in pipeline.named_steps else 'vect'
            with open('%s.%s.pkl' % (stats.prefix, clf.__class__.__name__.split('.')[-1]), 'wb') as outf:
                # pickle files needs to open in 'wb' mode
                logging.info('Pickling trained classifier to %s', outf.name)
                b = Bunch(clf=clf, inv_voc=inv_voc, tr_matrix=tr_matrix,
                          test_matrix=test_matrix, predictions=predictions,
                          y_tr=y_train, y_ev=y_test, train_mask=to_keep_train,
                          test_mask=to_keep_test)
                pickle.dump(b, outf)

        for metric, score in scores.items():
            scores_this_cv_run.append(
                [type(clf).__name__,
                 cv_i,
                 metric.split('.')[-1],
                 score])
        logging.info('Done with %s', clf)
    return scores_this_cv_run


def _analyze(scores, output_dir, name, class_names):
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
                                       '%s-%s' % (metric, class_names[id]),
                                       val])

    # save raw results
    df = DataFrame(cleaned_scores,
                   columns=['classifier', 'cv_no', 'metric', 'score'])
    csv = os.path.join(output_dir, '%s.out-raw.csv' % name)
    df.to_csv(csv, na_rep='-1')

    # now calculate mean and std
    grouped = df.groupby(['classifier', 'metric'])
    from numpy import mean, std

    # the mean and std columns may be appended in any order, put them in the desired order
    res = grouped['score'].aggregate({'score_mean': mean, 'score_std': std})[['score_mean', 'score_std']]

    # store csv for futher processing
    csv = os.path.join(output_dir, '%s.out.csv' % name)
    res.to_csv(csv, na_rep='-1')
    # df.to_excel(os.path.join(output_dir, '%s.out.xls' % name))

    return csv


def _config_logger(logs_dir=None, name='log', debug=False):
    if logs_dir and not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
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

    if logs_dir is not None:
        log_file = os.path.join(logs_dir, '%s.log' % name)
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
        shutil.rmtree(output)

    # **********************************
    # CREATE OUTPUT DIRECTORY
    # **********************************
    if not os.path.exists(output):
        logging.info('Creating output directory %s', glob(output))
        os.makedirs(output)


def go(conf_file, log_dir, data, vector_source, clean=False, n_jobs=1):
    config, configspec_file = parse_config_file(conf_file)

    logging.info('Reading configuration file from %s, conf spec from %s',
                 glob(conf_file)[0], configspec_file)
    output = config['output_dir']
    _prepare_output_directory(clean, output)
    shutil.copy(conf_file, output)

    # Runs all commands specified in the configuration file

    # **********************************
    # LOADING RAW TEXT
    # **********************************
    x_tr, y_tr, x_test, y_test = data

    # CREATE CROSSVALIDATION ITERATOR
    cv_iterator, y_vals = _build_crossvalidation_iterator(config['crossvalidation'],
                                                          y_tr, y_test)
    if x_test is not None:
        # concatenate all data, the CV iterator will make sure x_test is used for testing
        x_vals = list(x_tr)
        x_vals.extend(list(x_test))
    else:
        x_vals = x_tr

    all_scores = []
    score_func = ChainCallable(config['evaluation'])

    params = []
    for i, (train_idx, test_idx) in enumerate(cv_iterator):
        params.append((log_dir, config, i, score_func, test_idx, train_idx,
                       vector_source, x_vals, y_vals))

    scores_over_cv = Parallel(n_jobs=n_jobs)(delayed(_cv_loop)(*foo) for foo in params)
    all_scores.extend([score for one_set_of_scores in scores_over_cv for score in one_set_of_scores])
    class_names = dict(enumerate(sorted(set(y_vals))))
    output_file = _analyze(all_scores, config['output_dir'], config['name'], class_names)
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

    data, data_id, _ = load_text_data_into_memory(conf['training_data'], conf['test_data'])
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
    vector_store = get_thesaurus(conf)
    go(conf_file, log_dir, data, vector_store, classpath=classpath, clean=clean, n_jobs=1)