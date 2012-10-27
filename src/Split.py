'''
Created on Aug 13, 2012

@author: csg22

This script is used for taking the file path of a data file and then creating two separate files, one for training with 
the specified first n articles (and however many negatives) and the other for predicting using the remaining articles.
'''

import sys
import gzip

def split_data(input_filepath, output_directory, seen_data_cutoff, n_positives, run_num):
    
    train_file = '%s/train.%s.%s.%s.gz'%(output_directory, seen_data_cutoff, n_positives, run_num)
    predict_file = '%s/predict.%s.%s.%s.gz'%(output_directory, seen_data_cutoff, n_positives, run_num)

    gzipReader = gzip.open(input_filepath, 'rb')
    training = []
    predict = []
    foundPos = 0
    notAtEnd = True

    features = {}

    #read in the articles to be training data
    while (foundPos < int(n_positives)):
        temp = gzipReader.readline()
        tokens_counts = temp.split(' ')
        document_class = tokens_counts[0]
        tokens_counts = tokens_counts[1:]
        training.append('%s'%document_class)
        for token_count in tokens_counts:
            token,count = token_count.split(':')
            if token not in features:
                features[token] = len(features)
            training.append(' %i:%s'%(features[token], count))
        if (document_class == '1'):
            foundPos += 1
        if (len(temp) == 0):
            sys.exit('Not enough positive articles to train on!')



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
        
    return train_file, predict_file

if __name__ == '__main__':
    split_data(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10])