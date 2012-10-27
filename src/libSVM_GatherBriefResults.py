'''
Created on Aug 14, 2012

@author: csg22

Used by 'GatherResults', this script generates a file for each brief of an experiment (that has used the (libSVM) SVM classifier) which contains each of the runs (or folds) for the given brief.  
'''

from __future__ import division
import sys
import numpy as np
import csv
import os
import gzip

def main(article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, classifier, number_of_runs):
    
    briefs = ['burton','caudrilla','fco','irwinmitchell','kazakhmys','liberty','mcdonalds','medtronic','msconsumer','newzealand','npower','ocado','ppl','renault','savills','southafrica','sweden']
    number_of_runs = int(number_of_runs)
    homeFilepath = '/home/c/cs/csg22/'
    
    trueClassDir = '%s/FeatureSelectionTesting/DataSplits/beforeFS/'%homeFilepath
    testClassDir = '%s/FeatureSelectionTesting/ClassifierIO/SVMio/classifications/'%homeFilepath
    singleBriefConfmat = '%s/FeatureSelectionTesting/confmats/singleBrief/'%homeFilepath
    
    for brief in briefs:
        fileToOpen = '%s/%s.%s.%s.%s.%s.%s.%s.classifications.libsvm.1'%(testClassDir, brief, article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, classifier)
        if os.path.exists(fileToOpen):
            totalSoFar = 0
            resultsArray = np.empty(shape=(number_of_runs, 7), dtype = object)
            currentRunNum = 1
            while currentRunNum-1 < number_of_runs:
                trueNeg = 0
                truePos = 0 
                falsePos = 0
                falseNeg = 0
                trueClassReader = gzip.open('%s/%s.%s.%s.%s.%s.%s.%s.test.%s.gzip'%(trueClassDir, brief, article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, classifier, currentRunNum), 'rb')
                testClassReader = open('%s/%s.%s.%s.%s.%s.%s.%s.classifications.libsvm.%s'%(testClassDir, brief, article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, classifier, currentRunNum), 'r')
                
                for trueLine in trueClassReader:
                    predictedLine = testClassReader.readline()
                    true = trueLine.split(' ')
                    predicted = predictedLine.split(' ')
                    
                    if (predicted[0] == '-1.0'):
                        if (true[0] == '-1'):
                            trueNeg += 1
                        else:
                            falseNeg += 1
                    elif (predicted[0] == '1.0'):
                        if (true[0] == '1'):
                            truePos += 1
                        else:
                            falsePos += 1
                    elif (predicted[0] != ''):
                        print 'Detected End of File. (You should only see this once per file)'
                    else:
                        print 'There is a bug in the program if you can see this.'
                
                posRecall = truePos / (truePos + falseNeg)
                accuracy = (truePos + trueNeg)/(truePos + trueNeg + falsePos + falseNeg)
                totalSoFar += posRecall
                resultsArray[currentRunNum-1] = [currentRunNum, truePos, trueNeg, falsePos, falseNeg, accuracy, posRecall]    
                currentRunNum += 1
                
            #now print out the results to confmat file 
            first_row = ['runNo', 'tp', 'tn', 'fp', 'fn', 'accuracy', 'pos. Recall']
            
            csv_writer = csv.writer(open('%s/%s.%s.%s.%s.%s.%s.%s.csv'%(singleBriefConfmat, brief, article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, classifier), 'w'))
            csv_writer.writerow(first_row)
            csv_writer.writerows(resultsArray)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], int(sys.argv[7]))

