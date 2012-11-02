'''
Created on Oct 18, 2012

@author: ml249
'''
import os
import sys
import re


def read_libsvm_data(fh):
    """
    svm_read_problem(data_file_name) -> [y, x]

    Read LIBSVM-format data from data_file_name and return labels y
    and data instances x.
    """
    prob_y = []
    prob_x = []

    for line in fh:
        line = line.split(None, 1)
        # In case an instance with all zero features
        if len(line) == 1: line += ['']
        label, features = line
        xi = {}
        for _entropy in features.split():
            ind, val = _entropy.split(":")
            xi[int(ind)] = float(val)
        prob_y += [float(label)]
        prob_x += [xi]

    fh.close()
    return (prob_y, prob_x)

def train_fn_from_source(source_fn, output_dir, num_seen=None, fs=None, fc=None, stratified=False, job_id=1):
    source_fn,_,_ = source_fn.partition('.')
    _,source_fn = os.path.split(source_fn)
    out_fn = os.path.join(output_dir, 'train', '%s.train'%(source_fn))
    out_fn += '.%s'%(num_seen) if num_seen is not None else ''
    out_fn += '.stratified' if stratified else ''
    out_fn += '.%s'%(fs) if fs is not None else ''
    out_fn += '.%s'%(fc) if fc is not None else ''
    out_fn += '.%s'%(job_id)
    out_fn += '.gz'    
    return out_fn

def predict_fn_from_source(source_fn, output_dir, num_seen=None, fs=None, fc=None, stratified=False, job_id=1):
    source_fn,_,_ = source_fn.partition('.')
    _,source_fn = os.path.split(source_fn)
    out_fn = os.path.join(output_dir, 'predict', '%s.predict'%(source_fn))
    out_fn += '.%s'%(num_seen) if num_seen is not None else ''
    out_fn += '.stratified' if stratified else ''
    out_fn += '.%s'%(fs) if fs is not None else ''
    out_fn += '.%s'%(fc) if fc is not None else ''
    out_fn += '.%s'%(job_id)
    out_fn += '.gz'
    return out_fn

def model_fn_from_source(source_fn, output_dir, num_seen=None, fs=None, fc=None, classifier=None, stratified=False, job_id=1):
    source_fn,_,_ = source_fn.partition('.')
    _,source_fn = os.path.split(source_fn)
    out_fn = os.path.join(output_dir,'%s'%(source_fn))
    out_fn += '.%s'%(num_seen) if num_seen is not None else ''
    out_fn += '.stratified' if stratified else ''
    out_fn += '.%s'%(fs) if fs is not None else ''
    out_fn += '.%s'%(fc) if fc is not None else ''
    out_fn += '.%s.model'%classifier
    out_fn += '.%s'%(job_id)
    return out_fn

def find_mallet():
    mallet_exec = None
    for path in sys.path:
        if os.path.exists(os.path.join(path,'mallet')):
            mallet_exec = os.path.join(path,'mallet')
    
    if mallet_exec is None:
        raise RuntimeError('Can not find mallet executable. Please specify the '\
        'location of the mallet bin directory in classpath.')
    
    return mallet_exec