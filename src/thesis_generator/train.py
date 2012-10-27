'''
Created on Oct 25, 2012

@author: ml249
'''

import os
import sys
import gzip
import time

import ioutil
from __main__ import args

_num_seen = 200

def train_liblinear(source_file, output_dir, metric=None, fc=None):
    from liblinear import liblinearutil
        
    model_dir = os.path.join(output_dir, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    train_fn = ioutil.train_fn_from_source(source_file, output_dir,\
                                           num_seen=_num_seen,\
                                           fs=metric, fc=fc,\
                                           stratified=args.stratify)
    
    model_fn = ioutil.model_fn_from_source(source_file, model_dir,\
                                           num_seen=_num_seen,\
                                           fs=metric, fc=fc,\
                                           stratified=args.stratify,\
                                           classifier='liblinear')
    
    with gzip.open(train_fn, 'rb') as fh:
        train_y, train_x = ioutil.read_libsvm_data(fh)
    
    print 'Training \'liblinear\' - %s'%(time.strftime('%Y-%b-%d %H:%M:%S'))
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