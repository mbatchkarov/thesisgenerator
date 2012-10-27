'''
Created on Aug 13, 2012

@author: csg22
'''
import os

#to be run before experiments are carried out - sets up all the necessary directories - take care to run this in the right location
os.makedirs('FeatureSelectionTesting/DataSplits/beforeFS/')
os.makedirs('FeatureSelectionTesting/DataSplits/withFS/')

os.makedirs('FeatureSelectionTesting/ClassifierIO/MAXENTio/malletformatfiles/training/')
os.makedirs('FeatureSelectionTesting/ClassifierIO/MAXENTio/malletformatfiles/testing/')
os.makedirs('FeatureSelectionTesting/ClassifierIO/MAXENTio/malletformatfiles/classifiers/')

os.makedirs('FeatureSelectionTesting/ClassifierIO/NBio/malletformatfiles/training/')
os.makedirs('FeatureSelectionTesting/ClassifierIO/NBio/malletformatfiles/testing/')
os.makedirs('FeatureSelectionTesting/ClassifierIO/NBio/malletformatfiles/classifiers/')

os.makedirs('FeatureSelectionTesting/ClassifierIO/SVMio/models/')
os.makedirs('FeatureSelectionTesting/ClassifierIO/SVMio/classifications/')

os.makedirs('FeatureSelectionTesting/ClassifierIO/SVM_liblin_io/models/')
os.makedirs('FeatureSelectionTesting/ClassifierIO/SVM_liblin_io/classifications/')
os.makedirs('FeatureSelectionTesting/ClassifierIO/SVM_liblin_io/confmats/')

os.makedirs('FeatureSelectionTesting/confmats/singleBrief/')
os.makedirs('FeatureSelectionTesting/confmats/singleRun/')
os.makedirs('FeatureSelectionTesting/confmats/singleTestGroup/')