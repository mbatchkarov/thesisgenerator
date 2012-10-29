#!/usr/bin/python
'''
Created on Oct 18, 2012

@author: ml249
'''

import os
import sys
import shutil
#import subprocess
import gzip
import csv
import time
import glob
import numpy as np
try:
    import cPickle as pickle
except NameError:
    import pickle

import ioutil
import preprocess
import metrics
import config
import plotter


_cls_header = 'LABEL, PREDICTION, SCORE'
_num_seen = 200

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
                print '----> source file \'%s\''%(args.source)
                print '----> seen data %i'%(_num_seen)
                preprocess.split_data(in_fh, args.output, _num_seen)
        else:
            raise NotImplementedError('Reading non compressed files is \
                currently not supported.')
    else:
        # todo: handle the case where the source is a directory
        raise NotImplementedError('Reading input from directories is not \
            supported yet.')

def _stratify(args):
    train_in_fn = ioutil.train_fn_from_source(args.source, args.output, _num_seen, stratified=False)
    train_out_fn = ioutil.train_fn_from_source(args.source, args.output, _num_seen, stratified=True)
#    predict_fn = ioutil.predict_fn_from_source(args.source, args.output, _num_seen)
    
    with gzip.open(train_in_fn,'r') as input_fh,\
        gzip.open(train_out_fn,'w') as output_fh:
        preprocess.stratify(input_fh, output_fh, _num_seen)

#    preprocess.stratify(predict_fn)
    
# **********************************
# **********************************


# **********************************
# DO FEATURE SELECTION
# **********************************
def _feature_selection(args):
    if os.path.isfile(args.source):
        train_fn = ioutil.train_fn_from_source(args.source, args.output, _num_seen, stratified=args.stratify)
        predict_fn = ioutil.predict_fn_from_source(args.source, args.output, _num_seen) 
    else:
        pass
        # todo: handle the case where the source is a directory
    
    for metric in args.scoring_metric:
        print 'Perform feature selection - %s'%(time.strftime('%Y-%b-%d %H:%M:%S'))
        print '----> metric: %s'%metric
        print '----> feature count: %s'%args.feature_count
        
        with gzip.open(train_fn) as fh:
            features = preprocess.compute_feature_counts(fh)
        
        sorted_features = metrics.sort(features, metric)
        selected_features = set(sorted_features[:args.feature_count])
        
        train_out_fn = ioutil.train_fn_from_source(args.source,\
                                                       args.output,\
                                                       num_seen=_num_seen,\
                                                       fs=metric,\
                                                       fc=args.feature_count,\
                                                       stratified=args.stratify) 
        
        predict_out_fn = ioutil.predict_fn_from_source(args.source,\
                                                           args.output,\
                                                           num_seen=_num_seen,\
                                                           fs=metric,\
                                                           fc=args.feature_count)
        
        if not os.path.exists(os.path.split(train_out_fn)[0]):
            os.makedirs(os.path.split(train_out_fn)[0])
        
        if not os.path.exists(os.path.split(predict_out_fn)[0]):
            os.makedirs(os.path.split(predict_out_fn)[0])
        
        with gzip.open(train_fn, 'rb') as in_fh, \
                gzip.open(train_out_fn, 'wb') as out_fh:
            preprocess.strip_features(in_fh, out_fh, selected_features)
        
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
#    import predict
    for classifier in args.classifiers:
        if classifier == 'liblinear' or classifier == 'libsvm':
            if len(args.scoring_metric) > 0:
                for metric in args.scoring_metric:
                    try:
                        _svm_predict(args.source, args.output, \
                                           metric=metric,\
                                           fc=args.feature_count,\
                                           classifier=classifier)
                    except NameError as e:
                        print e
            else:
                _svm_predict(args.source, args.output,\
                                   classifier=classifier)

def _svm_predict(source_file, output_dir, metric=None, fc=None, classifier=None):
    from liblinear import liblinearutil
    from libsvm import svmutil
    
    predict_dir = os.path.join(output_dir, 'predict')
    if not os.path.exists(predict_dir):
        os.makedirs(predict_dir)
    
    predict_fn = ioutil.predict_fn_from_source(source_file, output_dir,\
                                               num_seen=_num_seen,\
                                               fs=metric,fc=fc)
    model_fn = ioutil.model_fn_from_source(source_file, \
                                           os.path.join(output_dir, 'models'),\
                                           num_seen=_num_seen, fs=metric,\
                                           fc=fc, classifier=classifier,\
                                           stratified=args.stratify)
    _,cls_fn = os.path.split(model_fn)
    cls_fn,_ = os.path.splitext(cls_fn)
    cls_fn = os.path.join(output_dir, 'classifications', '%s.predict.txt'%(cls_fn))
    if not os.path.exists( os.path.join(output_dir, 'classifications') ):
        os.makedirs( os.path.join(output_dir, 'classifications') )
    
    print 'Predict \'%s\' - %s'%(classifier, time.strftime('%Y-%b-%d %H:%M:%S'))
    print '----> predict data file: \'%s\''%predict_fn
    print '----> load model from: \'%s\''%model_fn
    print '----> write classifications to: \'%s\''%cls_fn
    
    if not os.path.exists(model_fn):
        raise NameError('Can not find model file \'%s\''%model_fn)
    
    if not os.path.exists(predict_fn):
        raise NameError('Can not find predict data file \'%s\''%predict_fn)
    
    with gzip.open(predict_fn, 'rb') as fh:
        predict_y, predict_x = ioutil.read_libsvm_data(fh)
    
    # silence libsvm
    devnull = open('/dev/null', 'w')
    oldstdout_fno = os.dup(sys.stdout.fileno())
    os.dup2(devnull.fileno(), 1)
    if classifier == 'liblinear':
        model = liblinearutil.load_model(model_fn)
        labels, _, vals = liblinearutil.predict(predict_y, predict_x, model,\
                                                args.prc_args)
    elif classifier == 'libsvm':
        model = svmutil.svm_load_model(model_fn)
        labels, _, vals = svmutil.svm_predict(predict_y, predict_x, model,\
                                              args.prc_args)
        
    os.dup2(oldstdout_fno, 1)
    
    with open(cls_fn, 'w') as fh:
        fh.write('%s\n'%_cls_header)
        for cls, label, val in zip(predict_y, labels, vals):
            # val is an array of scores, one for each class, although in the
            # current mode of running the framework it contans one value
            fh.write( '%1.0f, %1.0f, %1.4f\n'%(cls, label, val[0]) )
# **********************************
# **********************************


# **********************************
# CREATE CONFUSION MATRIX TABLES FOR
# VARYING THRESHOLDS
# **********************************
def _create_tables(args):
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
                reader.next()
                scores = []
                lines = []
                for cls, label, score in reader:
                    scores.append(float(score))
                    lines.append( (int(cls),int(label),float(score)) )
            
            scores = sorted(scores)
            num_thresholds = 20
            xs = np.linspace(min(scores), max(scores), num_thresholds)
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
if __name__ == '__main__':
    args = config.arg_parser.parse_args()

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
    
    # **********************************
    # SPLIT DATA
    # **********************************
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


#elif args.classifier.lower() == 'liblinear':
#    pass
#    io = '%s/FeatureSelectionTesting/ClassifierIO/SVM_liblin_io/'%home
#    
#    out = args.output
#        
#    #Train the classifier using train data
#    train_file = "%s/withFS/%s.%s.%s.%s.%s.%s.%s.train.%s"%(splitdata, brief, article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, classifier, run_num)
#    test_file = "%s/withFS/%s.%s.%s.%s.%s.%s.%s.test.%s"%(splitdata, brief, article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, classifier, run_num)
#    model_file = "%s/models/%s.%s.%s.%s.%s.%s.SVM_liblin.model.%s"%(io, brief, article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, run_num)
#    classifications_file = "%s/classifications/%s.%s.%s.%s.%s.%s.SVM_liblin.classifications.%s"%(io, brief, article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, run_num)
#    confmat_file = "%s/singleRun/%s.%s.%s.%s.%s.%s.%s.%s.csv"%(confmatDir, brief, article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, classifier, run_num)
#    #print test details
#    sys.stdout.write('Classifier: %s\nFeature Selection Method: %s\nBrief: %s\nSeen Data Cut-off: %s\nArticle Selection Method: %s\nNumber of Positives: %s\nFeature Cut-off: %s'%(classifier, feature_selection_method, brief, seen_data_cutoff, article_selection_method, n_positives, feature_cutoff))
#    sys.stdout.flush()
#    start_time = time()
#    
#    y, x = svm_read_problem(train_file)
#    model = train(y, x)
#    testLabels, a = svm_read_problem(test_file)
#    p_labs, p_acc, p_vals = predict(testLabels, a, model)
#    
#    counter = 0
#    tp = fp = tn = fn = 0
#    for predictedLabel in p_labs:
#        testLabel = testLabels[counter]
#        if testLabel == predictedLabel:
#            if testLabel == 1.0:
#                tp += 1
#            else:
#                tn += 1
#        else:
#            if testLabel == 1.0:
#                fn += 1
#            else:
#                fp += 1
#        counter += 1
#    
#    posRecall = tp / (tp + fn)
#    
#    first_row = ['tp', 'tn', 'fp', 'fn', 'Accuracy', 'Pos. Recall']
#    second_row = [tp, tn, fp, fn, ((tp+tn)/(tp + tn + fp + fn)), posRecall]
#    
#    csv_writer = csv.writer(open(confmat_file, 'w'))
#    csv_writer.writerow(first_row)
#    csv_writer.writerow(second_row)
#    
#    save_model(model_file, model)
#    
#    sys.stdout.write('Took %1.2f seconds to train and test.\n'%(time() - start_time))
#    sys.stdout.flush()

#elif classifier == 'MAXENT':
#    mallet = '%s/maxentfiles/bin/mallet'%home
#    io = '%sFeatureSelectionTesting/ClassifierIO/MAXENTio/'%home
#
#    #convert data split files into mallet files - the below just sets up the file names for them
#    train_mallet_file = "%s/malletformatfiles/training/%s.%s.%s.%s.%s.%s.%s.train.mallet.%s"%(io, brief, article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, classifier, run_num)
#    test_mallet_file = "%s/malletformatfiles/testing/%s.%s.%s.%s.%s.%s.%s.test.mallet.%s"%(io, brief, article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, classifier, run_num)
#    mallet_classifier_path = "%s/malletformatfiles/classifiers/%s.%s.%s.%s.%s.%s.MAXENT.classifier.%s"%(io, brief, article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, run_num)
#
#    #print test details
#    sys.stdout.write('Feature Selection Method: %s\nBrief: %s\nSeen Data Cut-off: %s\nArticle Selection Method: %s\nNumber of Positives: %s\nFeature Cut-off: %s'%(feature_selection_method, brief, seen_data_cutoff, article_selection_method, n_positives, feature_cutoff))
#    sys.stdout.flush()
#
#    featureSplitFilePathTrain = "%s/withFS/%s.%s.%s.%s.%s.%s.%s.train.%s"%(splitdata, brief, article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, classifier, run_num)
#    argslist = ["%s"%(mallet),"import-svmlight","--input","%s"%(featureSplitFilePathTrain),"--output", train_mallet_file]
#
#    start_time = time()
#    sys.stdout.write('mallet: %s\ntrainFile: %s\noutPut: %s'%(mallet,featureSplitFilePathTrain,train_mallet_file))
#    return_code = subprocess.call(argslist)
#    sys.stdout.write('Took %1.2f seconds to create mallet format train file.\n'%(time() - start_time))
#    sys.stdout.flush()
#
#    featureSplitFilePathTest = "%s/withFS/%s.%s.%s.%s.%s.%s.%s.test.%s"%(splitdata, brief, article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, classifier, run_num)
#    argslist = ["%s"%(mallet),"import-svmlight","--input","%s"%(featureSplitFilePathTest),"--use-pipe-from", train_mallet_file, "--output", test_mallet_file]
#
#    start_time = time()
#    return_code = subprocess.call(argslist)
#    sys.stdout.write('Took %1.2f seconds to create mallet format test file.\n'%(time() - start_time))
#    sys.stdout.flush()
#
#    #Train and test the classifier
#    #argslist = ["%s"%(mallet),"train-classifier","--training-file", train_mallet_file,"--testing-file", test_mallet_file, "--output-classifier", mallet_classifier_path, "--trainer", "MaxEnt"]
#    argslist = "%s train-classifier --training-file %s --testing-file %s --output-classifier %s --trainer MaxEnt"%(mallet, train_mallet_file, test_mallet_file, mallet_classifier_path)
#    start_time = time()
#    #return_code = subprocess.call(argslist)
#    return_code = os.popen(argslist).read()
#
#    sys.stdout.write("OUTPUT HERE:'%s'"%(return_code))
#    sys.stdout.flush()
#
#    sys.stdout.write('Took %1.2f seconds to train and test the classifier.\n'%(time() - start_time))
#    sys.stdout.flush()
#    mallet_CreateSingleRunResult.main(brief, article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, classifier, run_num, return_code)
#
#elif classifier == 'NB':
#    mallet = '%s/maxentfiles/bin/mallet'%home
#    io = '%s/FeatureSelectionTesting/ClassifierIO/NBio/'%home
#        
#    #convert data split files into mallet files - the below just sets up the file names for them
#    train_mallet_file = "%s/malletformatfiles/training/%s.%s.%s.%s.%s.%s.%s.train.mallet.%s"%(io, brief, article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, classifier, run_num)
#    test_mallet_file = "%s/malletformatfiles/testing/%s.%s.%s.%s.%s.%s.%s.test.mallet.%s"%(io, brief, article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, classifier, run_num)
#    mallet_classifier_path = "%s/malletformatfiles/classifiers/%s.%s.%s.%s.%s.%s.NB.classifier.%s"%(io, brief, article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, run_num)
#    
#    #print test details
#    sys.stdout.write('Feature Selection Method: %s\nBrief: %s\nSeen Data Cut-off: %s\nArticle Selection Method: %s\nNumber of Positives: %s\nFeature Cut-off: %s'%(feature_selection_method, brief, seen_data_cutoff, article_selection_method, n_positives, feature_cutoff))
#    sys.stdout.flush()
#    
#    featureSplitFilePathTrain = "%s/withFS/%s.%s.%s.%s.%s.%s.%s.train.%s"%(splitdata, brief, article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, classifier, run_num)
#    argslist = ["%s"%(mallet),"import-svmlight","--input","%s"%(featureSplitFilePathTrain),"--output", train_mallet_file]
#    
#    start_time = time()
#    return_code = subprocess.call(argslist)
#    sys.stdout.write('Took %1.2f seconds to create mallet format train file.\n'%(time() - start_time))
#    sys.stdout.flush()
#    
#    featureSplitFilePathTest = "%s/withFS/%s.%s.%s.%s.%s.%s.%s.test.%s"%(splitdata, brief, article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, classifier, run_num)
#    argslist = ["%s"%(mallet),"import-svmlight","--input","%s"%(featureSplitFilePathTest),"--use-pipe-from", train_mallet_file, "--output", test_mallet_file]
#    
#    start_time = time()
#    return_code = subprocess.call(argslist)
#    sys.stdout.write('Took %1.2f seconds to create mallet format test file.\n'%(time() - start_time))
#    sys.stdout.flush()
#    
#    #Train and test the classifier
#    #argslist = ["%s"%(mallet),"train-classifier","--training-file", train_mallet_file,"--testing-file", test_mallet_file, "--output-classifier", mallet_classifier_path]
#    argslist = "%s train-classifier --training-file %s --testing-file %s --output-classifier %s"%(mallet, train_mallet_file, test_mallet_file, mallet_classifier_path)
#    start_time = time()
#    #return_code = subprocess.call(argslist)
#    return_code = os.popen(argslist).read()
#    
#    sys.stdout.write("OUTPUT HERE:'%s'"%(return_code))
#    sys.stdout.flush()
#    
#    sys.stdout.write('Took %1.2f seconds to train and test the classifier.\n'%(time() - start_time))
#    sys.stdout.flush()
#    mallet_CreateSingleRunResult.main(brief, article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, classifier, run_num, return_code)
#    
#elif classifier == 'SVM_liblin': 

#    
#else:
#    sys.stdout.write('Invalid classifier given.')
#    sys.stdout.flush()
