'''
Created on Oct 17, 2012

@author: ml249
'''

import gzip


def extract_features(input_fh):
    global features
    global posCount
    global negCount
    
    features = {}
    posCount = 0
    negCount = 0
#    trainFileReader = gzip.open(input_file, 'rb')
    featuresOrderOfAppearance = []
    document_count = 0
    
    #process articles
    for currentArticle in input_fh:
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
            
            #if key doesn't exist add it to features dict:
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
    
    return features