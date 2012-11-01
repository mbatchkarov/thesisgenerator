'''
Created on Oct 17, 2012

@author: ml249
'''

import sys
import math
import random
import numpy as np


_min_float = sys.float_info.min
_max_float = sys.float_info.max

def sort(features_dict, metric_name):
    mod = sys.modules[__name__]
    symbols = vars(mod)
    if metric_name.lower() not in symbols:
        raise NameError('Can not find metric \'%s\''%metric_name)
    metric = symbols[metric_name.lower()]
    sorted_features = sorted( features_dict.keys(), key=lambda k: metric(features_dict[k]), reverse=True )
    return sorted_features

def _tpr(matrix):
    return matrix['tp'] / (matrix['tp']+matrix['fn']+0.)

def _fpr(matrix):
    return matrix['fp'] / (matrix['fp']+matrix['tn']+0.)

def acc(feature, *args):
    """Accuracy. tp - fp
    """
    tp = feature['tp']
    fp = feature['fp']
    feature['score'] = tp - fp
    return feature['score']
        
def acc2(feature, *args):
    """Balanced accuracy. abs(tpr - fpr)
    """
    feature['score'] = math.fabs(_tpr(feature) - _fpr(feature))
    return feature['score']
    
def bns(feature, *args):
    """Binormal Separation.
    
    Let F be the standard normal cumulative distribution function. Binormal
    separation is then abs(F^-1(tpr) - F^-1(fpr)), where F^-1 is the inverse
    c.d.f. of the standard normal.
    """
    tpr = _tpr(feature)
    fpr = _fpr(feature)
    
    try:
        invcdf_tpr = _ltqnorm(tpr)
    except ValueError:
        if tpr == 0: invcdf_tpr = _min_float # -inf
        elif tpr == 1: invcdf_tpr = _max_float # inf
    
    try:
        invcdf_fpr = _ltqnorm(fpr)
    except ValueError:
        if fpr == 0: invcdf_fpr = _min_float # -inf
        elif fpr == 1: invcdf_fpr = _max_float # inf
        
    feature['score'] = math.fabs(invcdf_tpr - invcdf_fpr)
    return feature['score']

def _chi2(count, expected):
    try:
        return (count - expected)**2 / (expected + 0.)
    except ZeroDivisionError:
        return 0
    
def chi2(feature, *args):
    """Chi squared.
    
    The metric assumes that the expected frequency of types is independent of
    class. The probability of a type occurring in a positive document is
    therefore the observed document frequency of that type multiplied by the
    probability of a positive document (as observed from the data).
    """
    tp = feature['tp']
    fp = feature['fp']
    tn = feature['tn']
    fn = feature['fn']
    Ppos = (tp + fn) / (tp + fp + tn +fn + 0.)
    Pneg = (fp + tn) / (tp + fp + tn +fn + 0.)
    
    feature['score'] = _chi2(tp, (tp + fp)*Ppos) + \
                        _chi2(fn, (fn + tn)*Ppos) + \
                        _chi2(fp, (tp + fp)*Pneg) + \
                        _chi2(tn, (fn + tn)*Pneg)
    return feature['score']
    
def dfreq(feature, *args):
    """Document frequency
    """
    tp = feature['tp']
    fp = feature['fp']
    feature['score'] = tp + fp
    return feature['score']
    
def f1(feature, *args):
    """F1 measure
    """
    tp = feature['tp']
    fp = feature['fp']
    fn = feature['fn']
    pos = tp + fn
    feature['score'] = (2 * tp) / (pos + tp + fp + 0.)

def entropy(*args):
    total = sum(args) + 0.
    
    # if an argument is 0 in the list of arguments set the score for that
    # component to 0
    e = map(lambda arg: 0 if arg == 0 else arg/total * np.log2(arg/total), args)
    return -sum(e)

def ig(feature, *args):
    """Information gain.
    
    Information gain is a measure for the decrease in entropy (or increase in
    certainty) if the document collection would be divided by feature. It is
    the difference between the entropy of the document collection and the sum of
    entropies of two bins, one containing a feature and the other not.  
    
    If a feature was found that perfectly separates the two classes then the
    entropies of the two bins would both be 0, making the information gain from
    that feature exactly equal to the entropy of the whole document collection.
    All uncertainty about the document collection with respect the classes would
    thus have been removed.    
    """
    tp = feature['tp']
    fp = feature['fp']
    tn = feature['tn']
    fn = feature['fn']
    pos = tp + fn
    neg = fp + tn
    Pword = (tp + fp) / (tp + fp + tn + fn)
    Pnotword = 1 - Pword
    
    feature['score'] = entropy(pos,neg) - \
        ( (Pword * entropy(tp, fp)) + (Pnotword * entropy(fn, tn)) )
    return feature['score']

def oddn(feature, *args):
    """Odds ratio numerator.
    
    tpr(1 - fpr)
    """
    tp = feature['tp']
    fp = feature['fp']
    tn = feature['tn']
    fn = feature['fn']
    pos = tp + fn
    neg = fp + tn
    tpr = tp / pos
    fpr = fp / neg
    feature['score'] = tpr * (1 - fpr)
    return feature['score']
    
def odds(feature, *args):
    """Odds ratio
    
    (tpr(1-fpr)) / ((1-tpr)fpr)
    """
    try:
#        feature['score'] = ( _tpr(feature)*(1-_fpr(feature)) ) /\
#                        ( (1-_tpr(feature))*_fpr(feature) )
        feature['score'] = feature['tp'] * feature['tn'] /\
                             (feature['fp'] * feature['fn'] + 0.)
    except ZeroDivisionError:
        feature['score'] = feature['tp'] * feature['tn']
        
    return feature['score']

def pow_k(feature, k=5):
    """Power. (1-fpr)^k - (1-tpr)^k
    
    *k* should be a positive integer or float.
    """
    feature['score'] = (1 - _fpr(feature))**k - (1 - _tpr(feature))**k
    return feature['score']
    
def pr(feature, *args):
    """Probability ratio
    """
    try:
        feature['score'] = _tpr(feature) / _fpr(feature)
    except ZeroDivisionError:
        feature['score'] = 1
    return feature['score']
    
def rand(feature, *args):
    #print 'Score metric specified: Random..'
    feature['score'] = random.random()
    return feature['score']

def _ltqnorm( p ):
    """
    Modified from the author's original perl code (original comments follow below)
    by dfield@yahoo-inc.com.  May 3, 2004.

    Lower tail quantile for standard normal distribution function.

    This function returns an approximation of the inverse cumulative
    standard normal distribution function.  I._entropy., given P, it returns
    an approximation to the X satisfying P = Pr{Z <= X} where Z is a
    random variable from the standard normal distribution.

    The algorithm uses a minimax approximation by rational functions
    and the result has a relative error whose absolute value is less
    than 1.15e-9.

    Author:      Peter John Acklam
    Time-stamp:  2000-07-19 18:26:14
    E-mail:      pjacklam@online.no
    WWW URL:     http://home.online.no/~pjacklam
    """

    if p <= 0 or p >= 1:
        # The original perl code exits here, we'll throw an exception instead
        raise ValueError( "Argument to _ltqnorm %f must be in open interval (0,1)" % p )

    # Coefficients in rational approximations.
    a = (-3.969683028665376e+01,  2.209460984245205e+02, \
         -2.759285104469687e+02,  1.383577518672690e+02, \
         -3.066479806614716e+01,  2.506628277459239e+00)
    b = (-5.447609879822406e+01,  1.615858368580409e+02, \
         -1.556989798598866e+02,  6.680131188771972e+01, \
         -1.328068155288572e+01 )
    c = (-7.784894002430293e-03, -3.223964580411365e-01, \
         -2.400758277161838e+00, -2.549732539343734e+00, \
          4.374664141464968e+00,  2.938163982698783e+00)
    d = ( 7.784695709041462e-03,  3.224671290700398e-01, \
          2.445134137142996e+00,  3.754408661907416e+00)

    # Define break-points.
    plow  = 0.02425
    phigh = 1 - plow

    # Rational approximation for lower region:
    if p < plow:
        q  = math.sqrt(-2*math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)

    # Rational approximation for upper region:
    if phigh < p:
        q  = math.sqrt(-2*math.log(1-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)

    # Rational approximation for central region:
    q = p - 0.5
    r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
