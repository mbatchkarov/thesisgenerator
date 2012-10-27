'''
Created on Aug 7, 2012

@author: csg22

This script takes in a training and testing file, carries out feature selection and creates new files (stored in the afterFS folder) to be used as the training and testing data for the classifiers. 
'''

from __future__ import division
import math
import random
import sys
import gzip

def main(input_training_file, input_testing_file, output_directory, feature_selection_method, feature_cutoff):
    
    global features
    global posCount
    global negCount
    
    features = {}
    posCount = 0
    negCount = 0
    trainFileReader = gzip.open(input_training_file, 'rb')
    featuresOrderOfAppearance = []
    document_count = 0
    
    #process articles
    for currentArticle in trainFileReader:
        if len(currentArticle.strip()) == 0:
            continue
        
        document_count += 1
        articleFeatureList = currentArticle.strip().split(" ")
        
        for feature in articleFeatureList[1:]:
            individualFeatureSplit = feature.split(":")
            #f_token,_,f_value = feature.partition(':')
            
            #if the key already exists:
            if individualFeatureSplit[0] in features:   
                #if article is positive:
                if articleFeatureList[0] == '1':
                    features[individualFeatureSplit[0]]['cfmat'][0] += 1 #increment tp
                else:
                    features[individualFeatureSplit[0]]['cfmat'][1] += 1 #increment fp
            
            #if key doesn'_chi2 exist add it to features dict:
            else:
                if articleFeatureList[0] == '1':
                    features[individualFeatureSplit[0]] = {"score":0,"cfmat":[1,0,posCount,negCount]}
                else:
                    features[individualFeatureSplit[0]] = {"score":0,"cfmat":[0,1,posCount,negCount]}
                featuresOrderOfAppearance.append(individualFeatureSplit[0])
        
        if articleFeatureList[0] == '1':
            posCount += 1
        else:
            negCount += 1
        
    for key in features:
        features[key]["cfmat"][2] = (posCount - features[key]["cfmat"][0]) #calculate tn
        features[key]["cfmat"][3] = (negCount - features[key]["cfmat"][1]) #calculate fn
        
    assert(sum(features[key]["cfmat"]) == document_count)
        
    if feature_selection_method != 'NoFS':
        #print 'Calculating feature scores...'
        possibles = globals().copy()
        possibles.update(locals())
        possibles.get(feature_selection_method)()
    
        #print 'Sorting features and applying cutoff...'
        sortedList = sorted(features.keys(), key=lambda k:features[k]["score"], reverse = True)
        selected_features = sortedList[:int(feature_cutoff)]
    
    elif feature_selection_method == 'NoFS':
        selected_features = featuresOrderOfAppearance[:int(feature_cutoff)]
        
    #print 'Writing new train file...'
    trainFile = '%s/%s.%s.train.gz'%(output_directory, feature_selection_method, feature_cutoff)
    trainFileWriter = gzip.open(trainFile, 'w')
    trainFileReader = gzip.open(input_training_file, 'rb')
    selected_features = set(selected_features)
    
    for article in trainFileReader:
        articleFeatureList = article.strip().split(" ")
        trainFileWriter.write("%s"%articleFeatureList[0])
        for feature in articleFeatureList[1:]:
            if feature.partition(":")[0] in selected_features:
                trainFileWriter.write(" %s"%feature)
        trainFileWriter.write("\n")
    
    #print 'Writing new test file...'
    testFile = '%s/%s.%s.test.gz'%(output_directory, feature_selection_method, feature_cutoff)
    testFileWriter = gzip.open(testFile, 'w')
    testFileReader = gzip.open(input_testing_file, 'rb')
    
    for article in testFileReader:
        articleFeatureList = article.strip().split(" ")
        testFileWriter.write("%s"%articleFeatureList[0])
        for feature in articleFeatureList[1:]:
            if feature.partition(":")[0] in selected_features:
                testFileWriter.write(" %s"%feature)
        testFileWriter.write("\n")
    
    #print 'Finished feature selection process.'
    return trainFile, testFile
    
    
def Acc():
    #print 'Score metric specified: Accuracy..'
    for key in features:
        tp = features[key]['cfmat'][0]
        fp = features[key]['cfmat'][1]
        features[key]['score'] = tp - fp
        
def Acc2():
    #print 'Score metric specified: Accuracy Balanced..'
    for key in features:
        tp = features[key]['cfmat'][0]
        fp = features[key]['cfmat'][1]
        tn = features[key]['cfmat'][2]
        fn = features[key]['cfmat'][3]
        pos = tp + fn
        neg = fp + tn
        tpr = tp / pos
        fpr = fp / neg
        features[key]['score'] = math.fabs(tpr - fpr)
    
def BNS():
    #print 'Score metric specified: Bi-Normal Seperation..'
    for key in features:
        tp = features[key]['cfmat'][0]
        fp = features[key]['cfmat'][1]
        tn = features[key]['cfmat'][2]
        fn = features[key]['cfmat'][3]
        pos = tp + fn
        neg = fp + tn
        tpr = tp / pos
        fpr = fp / neg
        
#        if tpr == 0:
#            invcdf_tpr = sys.float_info.min # -inf
#        elif tpr == 1:
#            invcdf_tpr = sys.float_info.max # inf
#        else:
#            invcdf_tpr = _ltqnorm(tpr)
#        
#        if fpr == 0:
#            invcdf_fpr = sys.float_info.min # -inf
#        elif fpr == 1:
#            invcdf_fpr = sys.float_info.max # inf
#        else:
#            invcdf_fpr = _ltqnorm(fpr)

        if tpr == 0:
            invcdf_tpr = 2.2250738585072014e-308 # -inf
        elif tpr == 1:
            invcdf_tpr = 1.7976931348623157e+308 # inf
        else:
            invcdf_tpr = _ltqnorm(tpr)
        
        if fpr == 0:
            invcdf_fpr = 2.2250738585072014e-308 # -inf
        elif fpr == 1:
            invcdf_fpr = 1.7976931348623157e+308 # inf
        else:
            invcdf_fpr = _ltqnorm(fpr)
            
        features[key]['score'] = math.fabs(invcdf_tpr - invcdf_fpr)
    
    
def Chi():
    #print 'Score metric specified: Chi-Squared..'
    for key in features:
        tp = features[key]['cfmat'][0]
        fp = features[key]['cfmat'][1]
        tn = features[key]['cfmat'][2]
        fn = features[key]['cfmat'][3]
        Ppos = (tp + fn) / (tp + fp + tn +fn)
        Pneg = (fp + tn) / (tp + fp + tn +fn)
        features[key]['score'] = _chi2(tp, (tp + fp)*Ppos) + _chi2(fn, (fn + tn)*Ppos) + _chi2(fp, (tp + fp)*Pneg) + _chi2(tn, (fn + tn)*Pneg)
    
def DFreq():
    #print 'Score metric specified: Document Frequency..'
    for key in features:
        tp = features[key]['cfmat'][0]
        fp = features[key]['cfmat'][1]
        features[key]['score'] = tp + fp
    
def F1():
    #print 'Score metric specified: F1 - Measure..'
    for key in features:
        tp = features[key]['cfmat'][0]
        fp = features[key]['cfmat'][1]
        fn = features[key]['cfmat'][3]
        pos = tp + fn
        features[key]['score'] = (2 * tp) / (pos + tp + fp)
    
def IG():
    #print 'Score metric specified: Information Gain..'
    for key in features:
        tp = features[key]['cfmat'][0]
        fp = features[key]['cfmat'][1]
        tn = features[key]['cfmat'][2]
        fn = features[key]['cfmat'][3]
        pos = tp + fn
        neg = fp + tn
        Pword = (tp + fp) / (tp + fp + tn + fn)
        Pnotword = 1 - Pword
        features[key]['score'] = _entropy(pos, neg) - ((Pword * _entropy(tp, fp)) + (Pnotword * _entropy(fn, tn)))

def OddN():
    #print 'Score metric specified: Odds Ratio Numerator..'
    for key in features:
        tp = features[key]['cfmat'][0]
        fp = features[key]['cfmat'][1]
        tn = features[key]['cfmat'][2]
        fn = features[key]['cfmat'][3]
        pos = tp + fn
        neg = fp + tn
        tpr = tp / pos
        fpr = fp / neg
        features[key]['score'] = tpr * (1 - fpr)
    
def Odds():
    #print 'Score metric specified: Odds Ratio..'
    for key in features:
        tp = features[key]['cfmat'][0]
        fp = features[key]['cfmat'][1]
        tn = features[key]['cfmat'][2]
        fn = features[key]['cfmat'][3]
        if (tp == 0 or tn == 0 or fp == 0 or fn == 0):
            features[key]['score'] = -1
        else:
            features[key]['score'] = (tp * tn) / (fp * fn)

def Pow():
    #print 'Score metric specified: Power..'
    k = 5
    for key in features:
        tp = features[key]['cfmat'][0]
        fp = features[key]['cfmat'][1]
        tn = features[key]['cfmat'][2]
        fn = features[key]['cfmat'][3]
        pos = tp + fn
        neg = fp + tn
        tpr = tp / pos
        fpr = fp / neg
        features[key]['score'] = math.pow((1 - fpr), k) - math.pow((1 - tpr), k)
    
def PR():
    #print 'Score metric specified: Probability Ratio..'
    for key in features:
        tp = features[key]['cfmat'][0]
        fp = features[key]['cfmat'][1]
        tn = features[key]['cfmat'][2]
        fn = features[key]['cfmat'][3]
        pos = tp + fn
        neg = fp + tn
        tpr = tp / pos
        fpr = fp / neg
        if(fp == 0):
            fpr = 0.0000000001 / neg
        features[key]['score'] = tpr / fpr
    
def Rand():
    #print 'Score metric specified: Random..'
    for key in features:
        features[key]['score'] = random.random()

def _chi2(count, expect):
    if expect == 0:
        expected = 0.000000001
    else:
        expected = expect
    return math.pow((count - expect), 2) / expected

def _entropy(x, y):
    if (x == 0 and y == 0):
        return 0
    elif (x == 0):
        return -(x / (x + y)) * 2.2250738585072014e-308 - (y / (x + y)) * math.log(y/(x + y), 2)
    elif (y == 0):
        return -(x / (x + y)) * math.log(x/(x + y), 2) - (y / (x + y)) * 2.2250738585072014e-308
    else:
        return -(x / (x + y)) * math.log(x/(x + y), 2) - (y / (x + y)) * math.log(y/(x + y), 2)

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

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10], sys.argv[11])
