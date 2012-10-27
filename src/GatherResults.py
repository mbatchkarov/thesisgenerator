'''
Created on Aug 30, 2012

@author: csg22

Run this script to gather the results for a single 'submit_feature_selection' execution. Each 'run' (or fold) for the briefs used in the experiment
will be compiled together in one file, and the overall results across all briefs will be created in another. 

Input should be the properties of the test which can uniquely identify it. Only run this once all experimenting has been completed.  
'''

import sys
import libSVM_GatherBriefResults
import GatherBriefResults
import CreateCollectedTestResults

article_selection_method = sys.argv[1]
seen_data_cutoff = sys.argv[2]
n_positives = sys.argv[3]
feature_selection_method = sys.argv[4] 
feature_cutoff = sys.argv[5]
classifier = sys.argv[6]
number_of_runs = sys.argv[7]

if classifier == 'SVM':
    libSVM_GatherBriefResults.main(article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, classifier, number_of_runs)
else:
    GatherBriefResults.main(article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, classifier, number_of_runs)
    
CreateCollectedTestResults.main(article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, classifier, number_of_runs)
