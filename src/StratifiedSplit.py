'''
Created on Aug 13, 2012

@author: csg22

This script takes the first n_positives and randomly selects 'neg_pos_data_count' positive and then negative articles to use for
training and then uses the rest of the data as prediction data. 
'''

import sys
import gzip
from random import choice 

def main(input_filepath, output_directory, classifier, brief, article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, run_num):

    #setup the file names/paths for the output  e.g. input /Volumes/LocalScratchHD/MLR/Data/libsvm/burton.libsvm.gzip
    train_file = '%s/%s.%s.%s.%s.%s.%s.%s.train.%s.gzip'%(output_directory, brief, article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, classifier, run_num)
    predict_file = '%s/%s.%s.%s.%s.%s.%s.%s.test.%s.gzip'%(output_directory, brief, article_selection_method, seen_data_cutoff, n_positives, feature_selection_method, feature_cutoff, classifier, run_num)

    gzipReader = gzip.open(input_filepath, 'rb')
    potentialTraining = [] #holds details about a single training article on the fly
    potentialTrainingCollection = [] #a collection of potential training articles (the above in single string form)
    predict = []
    foundPos = 0
    foundNeg = 0
    notAtEnd = True

    features = {}

    #read in the articles to form a potential training data pool
    while (foundPos < int(seen_data_cutoff)):
        potentialTraining = []
        temp = gzipReader.readline()
        tokens_counts = temp.split(' ')
        document_class = tokens_counts[0]
        tokens_counts = tokens_counts[1:]
        potentialTraining.append('%s'%document_class)
        for token_count in tokens_counts:
            token,count = token_count.split(':')
            if token not in features:
                features[token] = len(features)
            potentialTraining.append(' %i:%s'%(features[token], count))
        if (document_class == '1'):
            foundPos += 1
        else:
            foundNeg += 1
        if (len(temp) == 0):
            sys.exit('Not enough positive articles to train on.')
        potentialTrainingCollection.append(potentialTraining)

    if (foundNeg < int(seen_data_cutoff)):
            sys.exit('Not enough negative articles to train on.')

    #randomly choose the positive and negative articles to use and set these as the training data
    training = []
    chosenPos = 0
    chosenNeg = 0
    while(chosenPos < int(n_positives) or chosenNeg < int(n_positives)):
        randomArticle = choice(potentialTrainingCollection)
        if (randomArticle[0] == '1' and chosenPos < int(n_positives)):
            training.append(''.join(randomArticle))
            chosenPos += 1
        elif (randomArticle[0] == '-1' and chosenNeg < int(n_positives)):
            training.append(''.join(randomArticle))
            chosenNeg += 1
        potentialTrainingCollection.remove(randomArticle)

    #read in the articles to be predicted
    while (notAtEnd):
        temp = gzipReader.readline()
        if(len(temp) == 0):
            notAtEnd = False
        else:
            tokens_counts = temp.split(' ')
            document_class = tokens_counts[0]
            tokens_counts = tokens_counts[1:]
            predict.append('%s'%document_class)
            for token_count in tokens_counts:
                token,count = token_count.split(':')
                count = count.strip()
                if token in features:
                    predict.append(' %i:%s'%(features[token], count))
            predict.append('\n')

    #debugging - ensures that only one instance of each feature id exists in first training then second predict data
    temp1 = (''.join(training[1:])).split('\n')
    for line in temp1:
        idList = []
        temp2 = (''.join(line)).split(' ')
        temp2 = temp2[1:]
        for idColonFreq in temp2:
            idFreq = (''.join(idColonFreq)).split(':')
            idList.append(idFreq[0])
        idSet = set(idList)
        if(len(idList) != len(idSet)):
            sys.exit('There are duplicates in the training data!')

    temp1 = (''.join(predict[1:])).split('\n')
    for line in temp1:
        idList = []
        temp2 = (''.join(line)).split(' ')
        temp2 = temp2[1:]
        for idColonFreq in temp2:
            idFreq = (''.join(idColonFreq)).split(':')
            idList.append(idFreq[0])
        idSet = set(idList)
        if(len(idList) != len(idSet)):
            sys.exit('There are duplicates in the predict data!')

    output = gzip.open(train_file, 'wb')
    try:
        output.writelines(training)
    finally:
        output.close()

    output = gzip.open(predict_file, 'wb')
    try:
        output.writelines(predict)
    finally:
        output.close()

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10])
