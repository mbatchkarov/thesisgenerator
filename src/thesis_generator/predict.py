'''
Created on Oct 29, 2012

@author: mattilyra
'''

import os
import time
import gzip
import sys
import subprocess
import tempfile
import re

import ioutil
from __main__ import args


_cls_header = 'LABEL, PREDICTION, SCORE'
_num_seen = 200

def _output_paths(source_file, output_dir, metric=None, fc=None, classifier=None): 
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
    
    return model_fn, predict_fn, cls_fn

def svm_predict(source_file, output_dir, metric=None, fc=None, classifier=None):
    from liblinear import liblinearutil
    from libsvm import svmutil
    
    model_fn, predict_fn, cls_fn = _output_paths(source_file, output_dir,\
                                          metric, fc, classifier)
    
    
    if not os.path.exists( os.path.join(output_dir, 'classifications') ):
        os.makedirs( os.path.join(output_dir, 'classifications') )
    
    print 'Predict \'%s\' - %s'%(classifier, time.strftime('%Y-%b-%d %H:%M:%S'))
    print '--> predict data file: \'%s\''%predict_fn
    print '--> load model from: \'%s\''%model_fn
    print '--> write classifications to: \'%s\''%cls_fn
    
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
            # val is a single score or an array of probability estimates
            # depending on what arguments were passed to liblinear and/or libsvm
            val = reduce(lambda l,f: l + ['%1.4f'%f], val, [])
            fh.write( '%1.0f, %1.0f, %s\n'%(cls, label, val) )
            
def mallet_predict(source_file, output_dir, metric=None, fc=None, classifier=None):
    mallet_exec = ioutil.find_mallet()
    _,_,mallet_trainer = classifier.partition("_")
    model_fn, predict_fn, cls_fn = _output_paths(source_file, output_dir,\
                                         metric, fc, mallet_trainer)
    
    f_id, fpath = tempfile.mkstemp(suffix='.mallet', dir=args.output)
    
    print 'Predict \'%s\' - %s'%(mallet_trainer, time.strftime('%Y-%b-%d %H:%M:%S'))
    print '--> predict data file: \'%s\''%predict_fn
    print '--> load model from: \'%s\''%model_fn
    print '--> write classifications to: \'%s\''%cls_fn
    
    argslist = [mallet_exec,
                "classify-svmlight",
                "--input", fpath,
                "--output", cls_fn,
                "--classifier", model_fn]
    in_fh = gzip.open(predict_fn, 'r')
    os.write(f_id, in_fh.read())
    in_fh.close()
    
    ret_code = subprocess.call(argslist, stdin=subprocess.PIPE,\
                         stdout=subprocess.PIPE,\
                         stderr=subprocess.PIPE)
    
    os.close(f_id)
    os.unlink(fpath)
    
    # remove the leading garbage from the mallet classify file
    fh = open(cls_fn, 'r')
    f_id, fpath = tempfile.mkstemp(suffix='.mallet', dir=args.output)
    os.write(f_id, '%(_cls_header)s\n'%locals())
    with open(cls_fn, 'r') as predictions,\
        gzip.open(predict_fn, 'r') as true_labels:
        for predict, correct in zip(predictions, true_labels):
            predict = re.sub('^.*?\t','', predict)
            predict = re.sub('\t',', ', predict)
            true_cls,_,_ = correct.partition(' ')
            predict = '%s,%s'%(true_cls, predict)
            os.write(f_id, predict)
    
    fh.close()
    os.close(f_id)
    os.remove(cls_fn)
    os.rename(fpath, cls_fn)
    