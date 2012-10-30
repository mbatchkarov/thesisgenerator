'''
Created on Oct 25, 2012

@author: ml249
'''

import os
import sys
import gzip
import time
import subprocess
import tempfile

import ioutil
from __main__ import args

#_num_seen = 200

def _output_paths(metric=None, classifier=None):
    model_dir = os.path.join(args.output, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    train_fn = ioutil.train_fn_from_source(args.source, args.output,\
                                           num_seen=args.num_seen,\
                                           fs=metric, fc=args.feature_count,\
                                           stratified=args.stratify)
    
    model_fn = ioutil.model_fn_from_source(args.source, model_dir,\
                                           num_seen=args.num_seen,\
                                           fs=metric, fc=args.feature_count,\
                                           stratified=args.stratify,\
                                           classifier=classifier)

    return model_fn, train_fn

def train_svm(metric=None, classifier=None):
    from liblinear import liblinearutil
    from libsvm import svmutil
    
    model_fn, train_fn = _output_paths(metric, classifier)
    
    with gzip.open(train_fn, 'rb') as fh:
        train_y, train_x = ioutil.read_libsvm_data(fh)
        
    print 'Training \'%s\' - %s'%(classifier, time.strftime('%Y-%b-%d %H:%M:%S'))
    print '--> training data file: \'%s\''%train_fn
    print '--> save model to: \'%s\''%model_fn
    
    devnull = open('/dev/null', 'w')
    oldstdout_fno = os.dup(sys.stdout.fileno())
    os.dup2(devnull.fileno(), 1)
    if classifier == 'liblinear':
        model = liblinearutil.train(train_y, train_x, args.prc_args)
    elif classifier == 'libsvm':
        model = svmutil.svm_train(train_y, train_x, args.prc_args)
    os.dup2(oldstdout_fno, 1)
    
    if model_fn is not None and classifier == 'liblinear':
        liblinearutil.save_model(model_fn, model)
    elif model_fn is not None and classifier == 'libsvm':
        svmutil.svm_save_model(model_fn, model)
    
    return model

def _convert_to_mallet(mallet_exec, input_fpath):
    f_id, fpath = tempfile.mkstemp(suffix='.mallet', dir=args.output)
    argslist = [mallet_exec,
                "import-svmlight",
                "--input","-",
                "--output",fpath]
    
    p = subprocess.Popen(argslist, stdin=subprocess.PIPE,\
                         stdout=subprocess.PIPE,\
                         stderr=subprocess.PIPE)
    
    in_fh = gzip.open(input_fpath, 'r')
    out_data,err_data = p.communicate(in_fh.read())
    if len(out_data) > 0 or len(err_data) > 0:
        print '** Mallet Std Out **:\n', out_data, '*****************'
        print '** Mallet Std Err **:\n', err_data, '*****************' 
    in_fh.close()
    os.close(f_id)
    return f_id,fpath
    
def train_mallet(metric=None, classifier=None):
    # find the mallet executable
    mallet_exec = ioutil.find_mallet()
    _,_,mallet_trainer = classifier.partition("_")
    model_fn, train_fn = _output_paths(metric, mallet_trainer)
    
    # TODO: change the mallet data conversion so that it's done once per settings only
    print 'Convert data to Mallet format - %s'%(time.strftime('%Y-%b-%d %H:%M:%S'))
    _, mallet_train_fpath = _convert_to_mallet(mallet_exec, train_fn)
    
    print 'Training \'Mallet %s\' - %s'%(mallet_trainer, time.strftime('%Y-%b-%d %H:%M:%S'))
    print '--> training data file: \'%s\''%mallet_train_fpath
    print '--> save model to: \'%s\''%model_fn
    
    argslist = [mallet_exec,
                "train-classifier",
                "--report", "train:accuracy " # the space has to be there
                "--trainer", mallet_trainer,
                "--input",mallet_train_fpath,
                "--output-classifier",model_fn,
                "--verbosity", "0"]
    
    try:
#        fid_out, f_out = tempfile.mkstemp(suffix='.out.txt', dir=args.output)
#        fid_err, f_err = tempfile.mkstemp(suffix='.err.txt', dir=args.output)
        ret_code = subprocess.call(argslist, stdout=subprocess.PIPE)
    except subprocess.CalledProcessError as err:
        print 'Could not execute Mallet command, return code', err.returncode
        print err.output
    
    os.unlink(mallet_train_fpath)
    
