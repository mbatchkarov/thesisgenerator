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
import glob
import numpy as np
import multiprocessing as mp
from math import log

import ioutil
import preprocess
import metrics
import config
import plotter


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
        conf_fh.write( '***** %s *****\n********************************\n'\
                       %(time.strftime('%Y-%b-%d %H:%M:%S')) )
        for key in vars(args):
            conf_fh.write( '%s = %s\n'%(key, vars(args)[key]) )
        conf_fh.write( '************* END **************\n' )

# **********************************
# SPLIT DATA
# **********************************
def _split_data(args):
    if os.path.isfile(args.source):
        in_fh = open(args.source, 'rb')
        magic = in_fh.read(2)
        if magic == '\x1f\x8b':
            with gzip.open(args.source) as in_fh:
                print 'Split data - %s'%(time.strftime('%Y-%b-%d %H:%M:%S'))
                print '--> source file \'%s\''%(args.source)
                print '--> seen data %i'%(args.num_seen)
                preprocess.split_data(in_fh, args.output, args.num_seen)
        else:
            raise NotImplementedError('Reading non compressed files is '\
                'currently not supported.')
    else:
        # todo: handle the case where the source is a directory
        raise NotImplementedError('Reading input from directories is not '\
            'supported yet.')

def _stratify(args):
    train_in_fn = ioutil.train_fn_from_source(args.source, args.output,\
                                              args.num_seen, stratified=False)
    train_out_fn = ioutil.train_fn_from_source(args.source, args.output,\
                                               args.num_seen, stratified=True)
    
    with gzip.open(train_in_fn,'r') as input_fh,\
        gzip.open(train_out_fn,'w') as output_fh:
        preprocess.stratify(input_fh, output_fh, args.num_seen)
            
# **********************************
# **********************************


# **********************************
# DO FEATURE SELECTION
# **********************************
def _feature_selection(args):
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
    
    print 'Creating proces pool with %i processes'%(mp.cpu_count()+1)
    
    mp_pool = mp.Pool( processes = mp.cpu_count() )
    
    if not os.path.exists(os.path.join(args.output,'train')):
        os.makedirs(os.path.join(args.output,'train'))
    
    if not os.path.exists(os.path.join(args.output,'predict')):
        os.makedirs(os.path.join(args.output,'predict'))
    
    for metric in args.scoring_metric:
        mp_pool.apply_async(_select_features_using_metric,
                            args=(metric, train_fn,  predict_fn, features, args))
#        _select_features_using_metric(metric, train_fn,  predict_fn, features, args)
    mp_pool.close()
    mp_pool.join()

def _select_features_using_metric(metric, train_fn, predict_fn, features, args):
    print 'Perform feature selection - %s'%(time.strftime('%Y-%b-%d %H:%M:%S'))
    print '--> metric: %s'%metric
    print '--> feature count: %s'%args.feature_count
    
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
    
    with gzip.open(train_fn, 'rb') as in_fh, \
            gzip.open(train_out_fn, 'wb') as out_fh:
        if metric != 'none':
            preprocess.strip_features(in_fh, out_fh, selected_features)
        else:
            out_fh.write(in_fh.read())

    with gzip.open(predict_fn, 'rb') as in_fh, \
            gzip.open(predict_out_fn, 'wb') as out_fh:
        preprocess.strip_features(in_fh, out_fh, selected_features)
# **********************************
# **********************************


# **********************************
# TRAIN MODELS
# **********************************
def _train_models(args):
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
    import predict
    predict.args = args
    for classifier in args.classifiers:
        if classifier == 'liblinear' or classifier == 'libsvm':
            if len(args.scoring_metric) > 0:
                for metric in args.scoring_metric:
                    try:
                        predict.svm_predict(args.source, args.output, \
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
                        predict.mallet_predict(args.source, args.output, \
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
    import predict
    _tbl_header = 'THRESHOLD, TP, FP, TN, FN'
    
    # split the files paths of the classifier output so that the filenames of
    # output files are split into the settings used to produce the output
    files = glob.glob(os.path.join(args.create_tables, '*'))
    for i,f_path in enumerate(files):
        f_path, f_name = os.path.split(f_path)
        
        # each entry is a tuple of path_name and the settings that were used to
        # generate the results
        files[i] = tuple([os.path.join(f_path,f_name)] + f_name.split('.')[:-2])

    if not os.path.exists(os.path.join(args.output, 'tables')):
        os.makedirs(os.path.join(args.output, 'tables'))
    
    for settings in files:
        with open(os.path.join(args.output, 'tables',\
                               '.'.join(settings[1:]) + '.csv'), 'w') as out_fh:
            out_fh.write('%s\n'%_tbl_header)
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
                        lines.append( (int(cls),int(label),float(score)) )
                elif cls_header == predict._csv_header_confidence:
                    for cls, label, score, _ in reader:
                        # for computing the differencent classification
                        # boundaries the scores must be adjusted so that they
                        # are balances around 0
                        score = float(score) - .5
                        scores.append( score )
                        lines.append( (int(cls),int(label),score) )
                else:
                    raise RuntimeError('Unknown classifications file format: '\
                                       '"%s". Known formats\n\t"%s"\n\t"%s"'
                                       %(cls_header, predict._csv_header_scores,
                                         predict._csv_header_confidence) )
                            
            # try reclassifying all documents based on the scores found
            scores = sorted(scores)
            num_thresholds = 20
            xs = np.linspace(min(scores), max(scores), num_thresholds)
            
            # make sure that the default classifier behaviour (threshold == 0) 
            # is in the results table as well
            if 0 not in xs:
                xs = np.sort( np.concatenate((xs, np.array([0], dtype=np.float64))) )
            
            for threshold in xs:
                table = {'tp':0,'fp':0,'tn':0,'fn':0}
                for (cls,label,score) in lines:
                    label = 1 if score < threshold else -1
                    table = _update_table(table, cls, label)
                    
                writer.writerow(['%1.4f'%threshold,\
                                table['tp'],table['fp'],table['tn'], table['fn']])
# **********************************
# **********************************


# **********************************
# RUN THE LOT WHEN CALLED FROM THE
# COMMAND LINE
# **********************************
def run_tasks(args):
    # **********************************
    # CLEAN OUTPUT DIRECTORY
    # **********************************
    if args.clean and os.path.exists(args.output):
        shutil.rmtree(args.output)
    
    # **********************************
    # CREATE OUTPUT DIRECTORY
    # **********************************
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    _write_config_file(args)
    
    # **********************************
    # ADD classpath TO SYSTEM PATH
    # **********************************
    for path in args.classpath.split(os.pathsep):
        print 'Adding (%s) to system path'%glob.glob(path)
        sys.path.append(os.path.abspath(path))
    
#    print sys.path
    
    # **********************************
    # SPLIT DATA
    # **********************************
    # TODO if crossvalidate is true do this several times
    if args.split_data:
        _split_data(args)
    
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
        plotter.execute(args)

if __name__ == '__main__':
    args = config.arg_parser.parse_args()
    run_tasks(args)
    