'''
Created on Oct 18, 2012

@author: ml249
'''

import logging


logging.basicConfig(format='%(asctime)s %(module)s.%(funcName)s (line %(lineno)d) : %(message)s', datefmt='%d.%m.%Y %H:%M:%S')

def run_tasks(args):
    import __main__
    __main__.run_tasks(args)
    
# TODO add functionality so that experiments with different prc_args won't overwrite each other 
# TODO add functionality to do crossvalidation, this should be done with bootstrap resampling keeping the seen data portion separate from the future data
# TODO parallelise model training
# TODO parallelise model prediction
# TODO add a catch to the subprocess call so that error are reported