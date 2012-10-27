'''
Created on Aug 29, 2012

@author: csg22

This is used by the FeatureSelectionTest script to write a single confusion matrix for an experiment using one of the mallet classifiers (NB and MAXENT).
The single file confusion matrix will then be used later on to generate higher level results.  

Example of changing representations as far as analysing the confusion matrix is concerned:
(msconsumer)
Confusion Matrix, row=true, column=predicted  accuracy=0.8135391923990499
 label   0   1  |total
  0  1  38 101  |139
  1 -1  56 647  |703

(caudrilla)
Confusion Matrix, row=true, column=predicted  accuracy=0.9710365853658537
 label   0   1  |total
  0 -1 2548   .  |2548
  1  1  76   .  |76
'''

from __future__ import division
import sys
import csv
import re

def main(brief, article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, classifier, run_number, test_result_string):
    
    homeFilepath = '/home/c/cs/csg22/'
    #homeFilepath = '/Volumes/LocalScratchHD/LocalHome/csg22/'
    confmatFile = '%s/FeatureSelectionTesting/confmats/singleRun/'%homeFilepath
    
    test_result = test_result_string.replace('.', '0') #ensures that a '.' in the matrix is read as a 0 instead
    
    #pattern identifies the confusion matrix in the output from mallet framework
    matrixPattern = re.compile('label\s+0\s+1\s*\|total\s+0\s+(-1|1)\s+(\d+|.)\s+(\d+|.)\s*\|\d+\s+1\s+(1|-1)\s+(\d+|.)\s+(\d+|.)\s*\|\d+')
    matrixMatchObj = re.search(matrixPattern, test_result)
                
    #the mallet framework switches round the order which (1 and -1) and (1 and 0) match up. See the example at the top of the script.
    if(matrixMatchObj.group(1) == '-1'):
        tn = int(matrixMatchObj.group(2))
        fp = int(matrixMatchObj.group(3))
        fn = int(matrixMatchObj.group(5))
        tp = int(matrixMatchObj.group(6))
    else:
        tn = int(matrixMatchObj.group(6))
        fp = int(matrixMatchObj.group(5))
        fn = int(matrixMatchObj.group(3))
        tp = int(matrixMatchObj.group(2))
                
    posRecall = tp/(tp+fn)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    
    first_row = ['tp', 'tn', 'fp', 'fn', 'accuracy', 'pos. Recall']
    second_row = [tp, tn, fp, fn, accuracy, posRecall]
    
    csv_writer = csv.writer(open("%s/%s.%s.%s.%s.%s.%s.%s.%s.csv"%(confmatFile, brief, article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, classifier, run_number), 'w'))
    csv_writer.writerow(first_row)
    csv_writer.writerow(second_row)
    
if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], int(sys.argv[8]), sys.argv[9])