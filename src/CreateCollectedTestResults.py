'''
Created on Aug 28, 2012

@author: csg22

This script looks at each of the brief files for an experiment (where each of the runs have been combined into the single file) and generates a new file with the average statistics for each brief with the
variance and standard deviation of the positive recall for the runs of each brief. 
'''

from __future__ import division
import sys
import numpy as np
import csv
import os

def main(article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, classifier, number_of_runs):
    briefs = ['burton','caudrilla','fco','irwinmitchell','kazakhmys','liberty','mcdonalds','medtronic','msconsumer','newzealand','npower','ocado','ppl','renault','savills','southafrica','sweden']
    number_of_runs = int(number_of_runs)
    #homeFilePath = '/home/c/cs/csg22'
    homeFilePath = '/Volumes/LocalScratchHD/LocalHome/csg22/'
    singleBriefConfmat = '%s/FeatureSelectionTesting/confmats/singleBrief/'%homeFilePath
    testGroupConfmat = '%s/FeatureSelectionTesting/confmats/singleTestGroup/'%homeFilePath
    
    rowIndex = 0
    resultsArray = np.empty(shape=(len(briefs)+1, 9), dtype = object) #brief, tp, tn, fp, fn, accuracy, pos.recall, posRecallVar, posRecSD
    resultsArray[rowIndex] = ["Brief", "tp", "tn", "fp", "fn", "accuracy", "pos. recall", "posRec.Variance", "posRec.SD"]
    rowIndex += 1
    
    tempTpAvg = tempFpAvg = tempTnAvg = tempFnAvg = tempAcc = tempPosRecallAvg = 0
    for brief in briefs:
        fileToOpen = '%s/%s.%s.%s.%s.%s.%s.%s.csv'%(singleBriefConfmat, brief, article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, classifier)
        if os.path.exists(fileToOpen):
            singleRunReader = open(fileToOpen, 'r')
            singleRunReader.readline() #skip header
            runCounter = 0
            tempPosRecallArray = np.zeros(shape=(number_of_runs), dtype=np.float64)
            for row in singleRunReader:
                row = row.strip().split(',')
                runCounter += 1
                tempTpAvg += float(row[1])
                tempTnAvg += float(row[2])
                tempFpAvg += float(row[3])
                tempFnAvg += float(row[4])
                tempAcc += float(row[5])
                tempPosRecallAvg += float(row[6])
                tempPosRecallArray[runCounter - 1] = float(row[6])
            
            resultsArray[rowIndex] = [brief, tempTpAvg/runCounter, tempTnAvg/runCounter, tempFpAvg/runCounter, tempFnAvg/runCounter, round(tempAcc/runCounter, 5), round(tempPosRecallAvg/runCounter, 5), round(np.var(tempPosRecallArray, ddof=1), 5), round(np.std(tempPosRecallArray, ddof=1), 5)]
            tempTpAvg = tempFpAvg = tempTnAvg = tempFnAvg = tempAcc = tempPosRecallAvg = 0
        else:
            resultsArray[rowIndex] = [brief, '-', '-', '-', '-', '-', '-', '-', '-']
        rowIndex += 1
        
    csv_writer = csv.writer(open('%s/%s.%s.%s.%s.%s.%s.%s.csv'%(testGroupConfmat, article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, classifier, number_of_runs), 'w'))
    csv_writer.writerows(resultsArray)
    

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], int(sys.argv[7]))
    
    