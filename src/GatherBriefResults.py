'''
Created on Aug 28, 2012

@author: csg22

Used by 'GatherResults', this script generates a file for each brief of an experiment (that HASN'T used the (libSVM) SVM classifier) which contains each of the runs (or folds) for the given brief.  
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
    singleConfmat = '%s/FeatureSelectionTesting/confmats/singleRun/'%homeFilePath
    singleBriefConfmat = '%s/FeatureSelectionTesting/confmats/singleBrief/'%homeFilePath
    currentRunNum = 1
    
    for brief in briefs:
        fileToOpen = '%s/%s.%s.%s.%s.%s.%s.%s.1.csv'%(singleConfmat, brief, article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, classifier)
        if os.path.exists(fileToOpen):
            resultsArray = np.empty(shape=(number_of_runs+1, 7), dtype = object) #run num, tp, tn, fp, fn, accuracy, pos.recall
            resultsArray[0] = ["runNo", "tp", "tn", "fp", "fn", "accuracy", "pos. recall"]
            currentRunNum = 1
            while currentRunNum-1 < number_of_runs:
                currentFileReader = open('%s/%s.%s.%s.%s.%s.%s.%s.%s.csv'%(singleConfmat,brief,article_selection_method,seen_data_cutoff,n_positives,feature_selection_method,feature_cutoff,classifier,currentRunNum), 'r')
                currentFileReader.readline() #skip header
                singleRun = currentFileReader.readline().strip().split(',')
                resultsArray[currentRunNum] = [currentRunNum,singleRun[0],singleRun[1],singleRun[2],singleRun[3],singleRun[4],singleRun[5]]
                currentRunNum += 1
    
            csv_writer = csv.writer(open('%s/%s.%s.%s.%s.%s.%s.%s.csv'%(singleBriefConfmat, brief, article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, classifier), 'w'))
            csv_writer.writerows(resultsArray)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], int(sys.argv[7]))