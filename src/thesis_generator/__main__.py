#!/usr/bin/python
'''
Created on Oct 18, 2012

@author: ml249
'''

import os
import sys
import shutil
import gzip
import csv
import time
from glob import glob
import numpy as np
import multiprocessing as mp
import logging
import ast

import ioutil
import preprocess
import metrics
import config
import plotter

from joblib import Memory
from thesis_generator.utils import get_named_object

logger = logging.getLogger(__name__)

def _update_table(tbl, true, predicted):
    true = int(true)
    predicted = int(predicted)
    if true == 1:
        if predicted == 1: tbl['tp'] += 1
        elif predicted == -1: tbl['fn'] += 1
    elif true == -1:
        if predicted == 1: tbl['fp'] += 1
        elif predicted == -1: tbl['tn'] += 1
    return tbl

# **********************************
# **********************************


# **********************************
# WRITE CONFIG TO FILE
# **********************************
def _write_config_file(args):
    with open(os.path.join(args.output, 'conf.txt'), 'a+') as conf_fh:
        conf_fh.write('***** %s *****\n********************************\n'\
                      % (time.strftime('%Y-%b-%d %H:%M:%S')))
        for key in vars(args):
            conf_fh.write('%s = %s\n' % (key, vars(args)[key]))
        conf_fh.write('************* END **************\n')

# **********************************
# FEATURE EXTRACTION / PARSE
# **********************************
def feature_extract(**kwargs):
    def _filename_generator(file_list):
        for f in file_list:
            yield f

    def _content_generator(file_list):
        for f in file_list:
            with open(f,'rb') as fh:
                yield fh.read()
    
    # todo write docstring
    from sklearn.datasets import load_files
    import inspect

    vectorizer = get_named_object(kwargs['vectorizer'])

    # get the names of the arguments that the vectorizer takes 
    initialize_args = inspect.getargspec(vectorizer.__init__)[0]
    call_args = {arg: val for arg, val in kwargs.items() if\
                 val != '' and arg in initialize_args}

    # convert the arguments to their correct types
    for key in call_args:
        call_args[key] = ast.literal_eval(str(call_args[key]))

    # todo preprocessor, analyzer and tokenizer must be provided as part of the vectorizer
    # todo vocabulary needs to be a complex data type - this should be allowed to be a file reference
    # todo dtype should be expanded into a numpy type

    vectorizer = vectorizer(**call_args)

    if kwargs['input'] == '\'content\'' or kwargs['input'] == '':
        try:
            input_gen = kwargs['input_generator']
            source = kwargs['source'] 
            try:
                logger.info('Retrieving input generator for name '\
                            '%(input_gen)s'%locals())
                
                generator = get_named_object(input_gen)(kwargs['source'])
                targets = np.asarray([t for t in generator.targets()], dtype=np.int)
            except ImportError:
                logger.info('No input generator found for name '\
                            '\'%(input_gen)s\'. Using a file content '\
                            'generator with source \'%(source)s\''%locals())
                
                paths = glob( os.path.join(kwargs['source'], '*', '*') )
                generator = _content_generator(paths)
                targets = targets = load_files(input_gen).target
            
            term_freq_matrix = vectorizer.fit_transform(generator.documents())
        except KeyError:
            raise ValueError('Can not find a name for an input generator. When '\
                             'the input type for feature extraction is defined '\
                             'as content, an input_generator must also be '\
                             'defined. The defined input_generator should '\
                             'produce raw documents.')

    elif kwargs['input'] == '\'filename\'':
        input_files = glob( os.path.join(kwargs['source'], '*', '*') )
        targets = load_files(kwargs['source']).target
        term_freq_matrix = vectorizer.fit_transform(_filename_generator(input_files))
    elif kwargs['input'] == '\'file\'':
        raise NotImplementedError('The input type \'file\' is not supported yet.')
    else:
        raise NotImplementedError('The input type \'%s\' is not supported yet.' % kwargs['type'])

    return term_freq_matrix, targets

# **********************************
# SPLIT DATA
# **********************************
def _split_data(**kwargs):
    raise NotImplementedError('This action has not been ported to work with scikits.')
    if os.path.isfile(args.source):
        in_fh = open(args.source, 'rb')
        magic = in_fh.read(2)
        if magic == '\x1f\x8b':
            with gzip.open(args.source) as in_fh:
                print 'Split data - %s' % (time.strftime('%Y-%b-%d %H:%M:%S'))
                print '--> source file \'%s\'' % (args.source)
                print '--> seen data %i' % (args.num_seen)
                preprocess.split_data(in_fh, args.output, args.num_seen)
        else:
            raise NotImplementedError('Reading non compressed files is '\
                                      'currently not supported.')
    else:
        # todo: handle the case where the source is a directory
        raise NotImplementedError('Reading input from directories is not '\
                                  'supported yet.')


def _stratify(args):
    raise NotImplementedError('This action has not been ported to work with scikits.')
    train_in_fn = ioutil.train_fn_from_source(args.source, args.output,\
        args.num_seen, stratified=False)
    train_out_fn = ioutil.train_fn_from_source(args.source, args.output,\
        args.num_seen, stratified=True)

    with gzip.open(train_in_fn, 'r') as input_fh,\
    gzip.open(train_out_fn, 'w') as output_fh:
        preprocess.stratify(input_fh, output_fh, args.num_seen)

# **********************************
# **********************************


# **********************************
# DO FEATURE SELECTION
# **********************************
def _feature_selection(args):
    raise NotImplementedError('This action has not been ported to work with scikits.')
    if os.path.isfile(args.source):
        train_fn = ioutil.train_fn_from_source(args.source, args.output,\
            args.num_seen,\
            stratified=args.stratify)

        predict_fn = ioutil.predict_fn_from_source(args.source, args.output,\
            args.num_seen)
    else:
        pass
        # todo: handle the case where the source is a directory

    with gzip.open(train_fn) as fh:
        features = preprocess.compute_feature_counts(fh)
        features = dict(features)

    print 'Creating proces pool with %i processes' % (mp.cpu_count() + 1)

    mp_pool = mp.Pool(processes=mp.cpu_count())

    if not os.path.exists(os.path.join(args.output, 'train')):
        os.makedirs(os.path.join(args.output, 'train'))

    if not os.path.exists(os.path.join(args.output, 'predict')):
        os.makedirs(os.path.join(args.output, 'predict'))

    for metric in args.scoring_metric:
        mp_pool.apply_async(_select_features_using_metric,
            args=(metric, train_fn, predict_fn, features, args))
        #        _select_features_using_metric(metric, train_fn,  predict_fn, features, args)
    mp_pool.close()
    mp_pool.join()


def _select_features_using_metric(metric, train_fn, predict_fn, features, args):
    raise NotImplementedError('This action has not been ported to work with scikits.')
    print 'Perform feature selection - %s' % (time.strftime('%Y-%b-%d %H:%M:%S'))
    print '--> metric: %s' % metric
    print '--> feature count: %s' % args.feature_count

    if metric != 'none':
        sorted_features = metrics.sort(features, metric)
        selected_features = set(sorted_features[:args.feature_count])
    else:
        selected_features = set(features.keys())

    train_out_fn = ioutil.train_fn_from_source(args.source,\
        args.output,\
        num_seen=args.num_seen,\
        fs=metric,\
        fc=args.feature_count,\
        stratified=args.stratify)

    predict_out_fn = ioutil.predict_fn_from_source(args.source,\
        args.output,\
        num_seen=args.num_seen,\
        fs=metric,\
        fc=args.feature_count)

    with gzip.open(train_fn, 'rb') as in_fh,\
    gzip.open(train_out_fn, 'wb') as out_fh:
        if metric != 'none':
            preprocess.strip_features(in_fh, out_fh, selected_features)
        else:
            out_fh.write(in_fh.read())

    with gzip.open(predict_fn, 'rb') as in_fh,\
    gzip.open(predict_out_fn, 'wb') as out_fh:
        preprocess.strip_features(in_fh, out_fh, selected_features)

# **********************************
# **********************************


# **********************************
# TRAIN MODELS
# **********************************
def _train_models(args):
    raise NotImplementedError('This action has not been ported to work with scikits.')
    import train

    train.args = args
    for classifier in args.classifiers:
        if classifier == 'liblinear' or classifier == 'libsvm':
            if len(args.scoring_metric) > 0:
                for metric in args.scoring_metric:
                    train.train_svm(metric, classifier=classifier)
            else:
                train.train_svm(classifier=classifier)
        elif classifier.startswith('mallet'):
            if len(args.scoring_metric) > 0:
                for metric in args.scoring_metric:
                    train.train_mallet(metric, classifier=classifier)
            else:
                train.train_mallet(classifier=classifier)

# **********************************
# **********************************


# **********************************
# PERFORM PREDICTION
# **********************************
def _predict(args):
    raise NotImplementedError('This action has not been ported to work with scikits.')
    import predict

    predict.args = args

    if not os.path.exists(os.path.join(args.output, 'classifications')):
        os.makedirs(os.path.join(args.output, 'classifications'))

    for classifier in args.classifiers:
        if classifier == 'liblinear' or classifier == 'libsvm':
            if len(args.scoring_metric) > 0:
                for metric in args.scoring_metric:
                    try:
                        predict.svm_predict(args.source, args.output,\
                            metric=metric,\
                            fc=args.feature_count,\
                            classifier=classifier)
                    except NameError as e:
                        print e
            else:
                predict.svm_predict(args.source, args.output,\
                    classifier=classifier)
        if classifier.startswith('mallet'):
            if len(args.scoring_metric) > 0:
                for metric in args.scoring_metric:
                    try:
                        predict.mallet_predict(args.source, args.output,\
                            metric=metric,\
                            fc=args.feature_count,\
                            classifier=classifier)
                    except NameError as e:
                        print e
            else:
                predict.mallet_predict(args.source, args.output,\
                    classifier=classifier)

# **********************************
# **********************************


# **********************************
# CREATE CONFUSION MATRIX TABLES FOR
# VARYING THRESHOLDS
# **********************************
def _create_tables(args):
    raise NotImplementedError('This action has not been ported to work with scikits.')
    import predict

    _tbl_header = 'THRESHOLD, TP, FP, TN, FN'

    # split the files paths of the classifier output so that the filenames of
    # output files are split into the settings used to produce the output
    files = glob(os.path.join(args.create_tables, '*'))
    for i, f_path in enumerate(files):
        f_path, f_name = os.path.split(f_path)

        # each entry is a tuple of path_name and the settings that were used to
        # generate the results
        files[i] = tuple([os.path.join(f_path, f_name)] + f_name.split('.')[:-2])

    if not os.path.exists(os.path.join(args.output, 'tables')):
        os.makedirs(os.path.join(args.output, 'tables'))

    for settings in files:
        with open(os.path.join(args.output, 'tables',\
            '.'.join(settings[1:]) + '.csv'), 'w') as out_fh:
            out_fh.write('%s\n' % _tbl_header)
            writer = csv.writer(out_fh)

            with open(settings[0], 'r') as in_fh:
                reader = csv.reader(in_fh)
                cls_header = ','.join(reader.next())
                scores = []
                lines = []

                # there are two types of csv files, one written from
                # libsvm/liblinear which has three columns (label, prediction,
                # score) and one written out from mallet which has four columns
                # (label, prediction, p(c=0), p(c=1)). We don't actually care
                # what the confidence for the latter class is since the two
                # confidence values add to one
                if cls_header == predict._csv_header_scores:
                    for cls, label, score in reader:
                        scores.append(float(score))
                        lines.append((int(cls), int(label), float(score)))
                elif cls_header == predict._csv_header_confidence:
                    for cls, label, score, _ in reader:
                        # for computing the differencent classification
                        # boundaries the scores must be adjusted so that they
                        # are balances around 0
                        score = float(score) - .5
                        scores.append(score)
                        lines.append((int(cls), int(label), score))
                else:
                    raise RuntimeError('Unknown classifications file format: '\
                                       '"%s". Known formats\n\t"%s"\n\t"%s"'
                                       % (cls_header, predict._csv_header_scores,
                                          predict._csv_header_confidence))

            # try reclassifying all documents based on the scores found
            scores = sorted(scores)
            num_thresholds = 20
            xs = np.linspace(min(scores), max(scores), num_thresholds)

            # make sure that the default classifier behaviour (threshold == 0) 
            # is in the results table as well
            if 0 not in xs:
                xs = np.sort(np.concatenate((xs, np.array([0], dtype=np.float64))))

            for threshold in xs:
                table = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
                for (cls, label, score) in lines:
                    label = 1 if score < threshold else -1
                    table = _update_table(table, cls, label)

                writer.writerow(['%1.4f' % threshold,\
                                 table['tp'], table['fp'], table['tn'], table['fn']])

# **********************************
# **********************************


# **********************************
# RUN THE LOT WHEN CALLED FROM THE
# COMMAND LINE
# **********************************
def run_tasks(args, configuration):
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

        # TODO this needs to be redone after the scikits integration is complete
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
    actions = configuration.sections()

    # **********************************
    # FEATURE EXTRACTION
    # **********************************
    if 'feature_extraction' in actions and\
       bool(configuration.get('feature_extraction', 'run')):
        # make joblib cache the results of the feature extraction process
        # todo should figure out which values to ignore, currently use all (args + secion_options)
        mem_cache.cache(feature_extract)

        # create the keyword argument list the action should be run with, it is
        # very important that all relevant argument:value pairs are present
        # because joblib uses the hashed argument list to lookup cached results
        # of computations that have been executed previously
        options = dict(vars(args).items() + configuration.items('feature_extraction'))

        feature_extract(**options)


    # **********************************
    # SPLIT DATA
    # **********************************
    # TODO if crossvalidate is true do this several times
    if args.split_data:
        _split_data(vars(args))

        if args.stratify:
            _stratify(args)

    # **********************************
    # FEATURE SELECTION
    # **********************************
    if args.feature_selection and len(args.scoring_metric) > 0:
        _feature_selection(args)

    # **********************************
    # TRAIN MODELS
    # **********************************
    if args.train:
        _train_models(args)

    # **********************************
    # PREDICTION
    # **********************************    
    if args.predict:
        _predict(args)

    # **********************************
    # CREATE CONFUSION MATRIX TABLES FOR
    # VARYING THRESHOLDS
    # **********************************
    if args.create_tables is not None:
        _create_tables(args)

    # **********************************
    # CREATE PLOTS FROM CONFUSION MATRIX
    # TABLES
    # **********************************
    if args.create_figures is not None:
        raise NotImplementedError('This action has not been ported to work with scikits.')
        plotter.execute(args)

if __name__ == '__main__':
    # initialize the package, this is currently mainly used to configure the
    # logging framework

    args = config.arg_parser.parse_args()

    from ConfigParser import SafeConfigParser

    conf_parser = SafeConfigParser()

    logger.info('Reading configuration file from %s' % glob(args.configuration))
    conf_parser.read(args.configuration)

    run_tasks(args, conf_parser)
