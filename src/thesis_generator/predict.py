'''
Created on Oct 29, 2012

@author: mattilyra
'''

import os
import time
import gzip
import sys
import subprocess
import multiprocessing as mp
import tempfile
import re
import warnings

import ioutil


args = None
_csv_header_scores = 'LABEL, PREDICTION, SCORE'
_csv_header_confidence = 'LABEL, PREDICTION, PROB(c0), PROB(c1)'

def _output_paths(source_file, output_dir, metric=None, fc=None, classifier=None): 
    predict_dir = os.path.join(output_dir, 'predict')
    if not os.path.exists(predict_dir):
        os.makedirs(predict_dir)
    
    predict_fn = ioutil.predict_fn_from_source(source_file, output_dir,\
                                               num_seen=args.num_seen,\
                                               fs=metric,fc=fc)
    
    model_fn = ioutil.model_fn_from_source(source_file, \
                                           os.path.join(output_dir, 'models'),\
                                           num_seen=args.num_seen, fs=metric,\
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
    
    print 'Predict \'%s\' - %s'%(classifier, time.strftime('%Y-%b-%d %H:%M:%S'))
    print '--> predict data file: \'%s\''%predict_fn
    print '--> load model from: \'%s\''%model_fn
    print '--> write classifications to: \'%s\''%cls_fn
    print '----> process arguments: \'%s\''%args.prc_args
    
    if not os.path.exists(model_fn):
        raise NameError('Can not find model file \'%s\''%model_fn)
    
    if not os.path.exists(predict_fn):
        raise NameError('Can not find predict data file \'%s\''%predict_fn)
    
    with gzip.open(predict_fn, 'rb') as fh:
        predict_y, predict_x = ioutil.read_libsvm_data(fh)
    
    # the only thing that can be in the process arguments for predictions is
    # '-b 1' anything else will throw an error
    argv = args.prc_args.split()
    i = 0
    prc_args = []
    while i < len(argv):
        if argv[i] == '-b':
            prc_args.append(' '.join([argv[i], argv[i+1]]))
            i += 2
        else:
            warn_msg = 'LibSVM / LibLinear does not allow any other command '\
                'line arguments for prediction than  \'-b {0,1}\'. Removing '\
                'argument \'%s\''%(argv[i])
            warnings.warn(warn_msg)
            i += 1
    prc_args = ' '.join(prc_args)

    # because liblinear and libsvm seem to segfaul every now and then the
    # training is best done in a separate process that can be restarted if the
    # svm library segfaults 
    def _run_prediction(classifier, predict_y, precit_x, model, argv, model_fn, q):
        if classifier == 'liblinear':
            labels, vals = liblinearutil.predict(predict_y, predict_x, model, argv)
        elif classifier == 'libsvm':
            labels, vals = svmutil.svm_predict(predict_y, predict_x, model, argv)
        
        q.put(labels)
        q.put(vals)

#    if classifier == 'liblinear':
#        model = liblinearutil.load_model(model_fn)
#        labels, vals = liblinearutil.predict(predict_y, predict_x, model,\
#                                                prc_args)
#    elif classifier == 'libsvm':
#        model = svmutil.svm_load_model(model_fn)
#        labels, vals = svmutil.svm_predict(predict_y, predict_x, model,\
#                                              prc_args)
    
    q = mp.Queue(3)
    while q.empty():
        print 'Running svm prediction in a subprocess', q.empty()
        p = mp.Process(target=_run_prediction, args=(classifier, predict_y, predict_x, argv, model_fn, q))
        p.start()
        p.join()
        
    labels = q.get()
    vals = q.get()
    
    with open(cls_fn, 'w') as fh:
        fh.write('%s\n'%_csv_header_scores)
        
        for cls, label, val in zip(predict_y, labels, vals):
            # val is a single score or an array of probability estimates
            # depending on what arguments were passed to liblinear and/or libsvm
            val = reduce(lambda l,f: l + ['%1.4f'%f], val, [])
            val = ','.join(val)
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
    print '----> process arguments: \'%s\''%args.prc_args
    
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
    f_id, fpath = tempfile.mkstemp(prefix='.', suffix='.mallet', dir=args.output)
    os.write(f_id, '%s\n'%_csv_header_confidence)
    with open(cls_fn, 'r') as predictions,\
        gzip.open(predict_fn, 'r') as true_labels:
        for predict, correct in zip(predictions, true_labels):
            predict = re.sub('^.*?\t','', predict.strip())
            cols = predict.split('\t')
            true_cls,_,_ = correct.strip().partition(' ')
            predict_cls = 1 if float(float(cols[1])) > float(float(cols[3])) else -1 
            predict = '%s, %i, %1.4f, %1.4f\n'%(true_cls, predict_cls,\
                                                float(cols[1]), float(cols[3]))
            os.write(f_id, predict)
    
    fh.close()
    os.close(f_id)
    os.remove(cls_fn)
    os.rename(fpath, cls_fn)
    