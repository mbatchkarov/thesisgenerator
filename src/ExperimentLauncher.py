#!/usr/bin/python

#from __future__ import division
import os
import sys
import shutil
import subprocess
import gzip
from time import time
import csv
import argparse
import time
try:
    import cPickle as pickle
except NameError:
    import pickle

import ioutil
import preprocess
import metrics
    

# **********************************
# SETUP ARGUMENTS PARSER
# **********************************

arg_parser = argparse.ArgumentParser(description='Launch an experiment.')

action_group = arg_parser.add_argument_group()
action_group.add_argument('--split-data', help='Split a data file into \
                training and test data.', action='store_true')
action_group.add_argument('--feature-selection', help='Perform feature \
                selection on input train and test files.', action='store_true')
action_group.add_argument('--train', help='Train a model. To train a model\
                the --feature-selector and --feature-count have to defined\
                so that the correct input files can be read into the training.',
                action='store_true', default=False)
action_group.add_argument('--predict', help='Evaluate a model on the data \
                in --evaluate-file', action='store_true', default=False)
action_group.add_argument('--clean', help='Clean the output directory of all \
                files before running any other commands',
                action='store_true', default=False)

arg_parser.add_argument('-id', '--jobid',
                        help='A numerical id for the job. This is used as a \
                            prefix to output files.',
                        type=int,
                        default=1,
                        metavar='1')

arg_parser.add_argument('-o', '--output',
                        help='Output directory.',
                        type=str,
                        required=True,
                        metavar='OUTPUT_DIR')

arg_parser.add_argument('-s', '--source',
                        help='Input file or directory for raw data when \
                        performing the data split. When performing feature \
                        selection or classifier training evaluation the file \
                        name(s) in this variable are used to find the \
                        intermediary that contain train/test data.',
                        type=str,
                        metavar='INPUT_DIR/FILE',
                        required=True)

arg_parser.add_argument('-cp', '--classpath',
                        help='Classpath for searching binaries. If several \
                        directories are provided they should be separated \
                        using the system path separator (\':\' on *nix)',
                        default='.')

split_data_group = arg_parser.add_argument_group('Splitting data')
split_data_group.add_argument('--stratify',
                        help='Stratify training data.',
                        action='store_true',
                        default=False)

split_data_group.add_argument('--seen-data-cutoff',
                              help='How many positive articles should be \
                                  considered to be the seen data from which \
                                  the training data is sampled.',
                              type=int,
                              default=200,
                              metavar='200')

split_data_group.add_argument('--train-data-size',
                        help='The number of positive documents to add to the \
                            training data.',
                        type=int,
                        default=200,
                        metavar='200')

feature_selection_group = arg_parser.add_argument_group('Performing feature \
                                                        selection')
feature_selection_group.add_argument('-sm', '--scoring-metric',
                        help='Feature selection scoring metric to be used to \
                            order the features found in the training data and \
                            prune the training and testing files. Test files \
                            are limited to the features seen in the training \
                            data.',
                        type=str,
                        choices=['rand','acc','acc2','bns','chi2','dfreq',
                                 'f1','ig','oddn','odd','pr','pow'],
                        default=[],
                        nargs='+')

feature_selection_group.add_argument('-fc', '--feature-count',
                        help='Number of features to be selected from the \
                            training data.',
                        type=int,
                        default=None,
                        metavar='8000')

train_group = arg_parser.add_argument_group('Training and evaluating classifier(s)')
train_group.add_argument('-c', '--classifiers',
                        help='Which classifier(s) should be trained and tested.',
                        type=str,
                        nargs='+',
                        choices=['libsvm','liblinear',
                                 'mallet_maxent','mallet_nb'])

train_group.add_argument('--crossvalidate',
                        help='Perform crossvalidation.',
                        action='store_true')

train_group.add_argument('--prc-args',
                        help='A space separated string of arguments to be \
                            passed to the subprocess running the classifiers. \
                            At a minimum the location of the target executable \
                            has to be specified. Please refer to the \
                            documentation of the respective libraries used to \
                            classify documents to see how the library can be \
                            configured.',
                        type=str,
                        default='')

#train_group.add_argument('--train-file',
#                        help='The file the training data should be read from.',
#                        type=str,
#                        default=None)
#
#train_group.add_argument('--predict-file',
#                        help='The file the prediction data should be read from.',
#                        type=str,
#                        default=None)

#train_group.add_argument('--model-file',
#                        help='The file(s) the trained model should be written \
#                        to and read from. These should be in the same order as \
#                        the classifiers to be trained.',
#                        type=str,
#                        nargs='+',
#                        default=[])

args = arg_parser.parse_args()

# **********************************
# **********************************


# **********************************
# CLEAN OUTPUT DIRECTORY
# **********************************

if args.clean and os.path.exists(args.output):
    shutil.rmtree(args.output)

# **********************************
# **********************************


# **********************************
# CREATE OUTPUT DIRECTORY
# **********************************

if not os.path.exists(args.output):
    os.makedirs(args.output)

# **********************************
# **********************************


# **********************************
# WRITE CONFIG TO FILE
# **********************************

with open(os.path.join(args.output, 'conf.txt'), 'a+') as conf_fh:
    conf_fh.write( '***** %s *****\n********************************\n'\
                   %(time.strftime('%Y-%b-%d %H:%M:%S')) )
    for key in vars(args):
        conf_fh.write( '%s = %s\n'%(key, vars(args)[key]) )
    conf_fh.write( '************* END **************\n' )

# **********************************
# **********************************


# **********************************
# ADD classpath TO SYSTEM PATH
# **********************************

for path in args.classpath.split(os.pathsep):
    print 'Adding (%s) to system path'%path
    sys.path.append(os.path.abspath(path))

# **********************************
# **********************************


# **********************************
# SPLIT DATA
# **********************************

if args.split_data:
    if os.path.isfile(args.source):
        in_fh = open(args.source, 'rb')
        magic = in_fh.read(2)
        if magic == '\x1f\x8b':
            with gzip.open(args.source) as in_fh:
                print 'Split data - %s'%(time.strftime('%Y-%b-%d %H:%M:%S'))
                print '----> source file \'%s\''%(args.source)
                print '----> seen data %i'%(200)
                train_fn, predict_fn = preprocess.split_data(in_fh, args.output,\
                                                             200)
    else:
        pass
        # todo: handle the case where the source is a directory

# **********************************
# **********************************


# **********************************
# DO FEATURE SELECTION
# **********************************

if args.feature_selection and len(args.scoring_metric) > 0:
    if os.path.isfile(args.source):
        train_fn = ioutil.train_fn_from_source(args.source, args.output)
        predict_fn = ioutil.predict_fn_from_source(args.source, args.output) 
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
        
        train_out_fn = ioutil.train_fn_from_source(args.source, \
                                                       args.output, metric, \
                                                       args.feature_count) 
        
        predict_out_fn = ioutil.predict_fn_from_source(args.source, \
                                                           args.output, metric,\
                                                           args.feature_count)
        
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

#def _match_classifiers_to_model_files(args):
#    if len(args.classifiers) > len(args.model_file):
#        args.model_file += [None]*( len(args.classifiers) - len(args.model_file) )
#    elif len(args.classifiers) < len(args.model_file):
#        args.classifiers += [None]*( len(args.model_file) - len(args.classifiers) )
#    
#    return args

def _liblinear_train(source_file, output_dir, metric=None, fc=None):    
    model_dir = os.path.join(output_dir, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    train_fn = ioutil.train_fn_from_source(source_file, output_dir, metric, fc)
    model_fn = ioutil.model_fn_from_source(source_file, model_dir, metric, \
                                             fc, 'liblinear')
    
    with gzip.open(train_fn, 'rb') as fh:
        train_y, train_x = ioutil.read_libsvm_data(fh)
    
    print 'Training \'%s\' - %s'%(classifier, time.strftime('%Y-%b-%d %H:%M:%S'))
    print '----> training data file: \'%s\''%train_fn
    print '----> save model to: \'%s\''%model_fn 
    
    # silence libsvm
    devnull = open('/dev/null', 'w')
    oldstdout_fno = os.dup(sys.stdout.fileno())
    os.dup2(devnull.fileno(), 1)
    model = liblinearutil.train(train_y, train_x, args.prc_args)
    os.dup2(oldstdout_fno, 1)
    
    if model_fn is not None:
        liblinearutil.save_model(model_fn, model)
    return model

def _liblinear_predict(source_file, output_dir, metric=None, fc=None):    
    predict_dir = os.path.join(output_dir, 'predict')
    if not os.path.exists(predict_dir):
        os.makedirs(predict_dir)
    
    predict_fn = ioutil.predict_fn_from_source(source_file, output_dir, metric,\
                                               fc)
    model_fn = ioutil.model_fn_from_source(source_file, \
                                           os.path.join(output_dir, 'models'),\
                                           metric, fc, 'liblinear')
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
        raise NameError('Can not find predict data file \'%s\''%model_fn)
    
    with gzip.open(predict_fn, 'rb') as fh:
        predict_y, predict_x = ioutil.read_libsvm_data(fh)
    
    # silence libsvm
    devnull = open('/dev/null', 'w')
    oldstdout_fno = os.dup(sys.stdout.fileno())
    os.dup2(devnull.fileno(), 1)
    model = liblinearutil.load_model(model_fn)
    labels, acc, vals = liblinearutil.predict(predict_y, predict_x, model, args.prc_args)
    os.dup2(oldstdout_fno, 1)
    
    with open(cls_fn, 'w') as fh:
        for cls, label, val in zip(predict_y, labels, vals):
            val = ','.join( map(lambda _entropy: '%1.4f'%_entropy, val) )
            fh.write('%1.0f, %1.0f, %s\n'%(cls, label, val))

if args.train:
    for classifier in args.classifiers:
        if classifier == 'liblinear':
            from liblinear import liblinearutil
            
            if len(args.scoring_metric) > 0:
                for metric in args.scoring_metric:
                    _liblinear_train(args.source, args.output,\
                                     metric, args.feature_count)
            else:
                _liblinear_train(args.source, args.output)
            
# **********************************
# **********************************


# **********************************
# PERFORM PREDICTION
# **********************************

if args.predict:
#    args = _match_classifiers_to_model_files(args)
#    assert( len(args.classifiers) == len(args.model_file) )
        
    for classifier in args.classifiers:
        if classifier == 'liblinear':
            from liblinear import liblinearutil
            
            if len(args.scoring_metric) > 0:
                for metric in args.scoring_metric:
                    try:
                        _liblinear_predict(args.source, args.output, \
                                           metric, args.feature_count)
                    except NameError as e:
                        print e
            else:
                _liblinear_predict(args.source, args.output)


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
