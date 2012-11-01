'''
Created on Oct 16, 2012

@author: ml249
'''

import os
import gzip
import re
import string
import collections
import random
try:
    from xml.etree import cElementTree as ET
except ImportError:
    from xml.etree import ElementTree as ET

import ioutil


def split_data(input_fh, output_dir, num_seen=200):
    train_fn = ioutil.train_fn_from_source(input_fh.name, output_dir, num_seen=num_seen)
    predict_fn = ioutil.predict_fn_from_source(input_fh.name, output_dir, num_seen=num_seen)
    
    if not os.path.exists(os.path.split(train_fn)[0]):
        os.makedirs(os.path.split(train_fn)[0])
    
    if not os.path.exists(os.path.split(predict_fn)[0]):
        os.makedirs(os.path.split(predict_fn)[0])
    
    with gzip.open(train_fn, 'wb') as train_fh, \
            gzip.open(predict_fn, 'wb') as predict_fh:
        pos = 0
        for line in input_fh:
            if pos < num_seen:
                train_fh.write(line)
            else:
                predict_fh.write(line)
            doc_class,_,_ = line.partition(' ')
            pos += 1 if int(doc_class) == 1 else 0

def stratify(input_fh, output_fh, num_seen):
    pos = []
    neg = []
    for i,line in enumerate(input_fh):
        line_tuple = line.partition(' ')
        if line_tuple[0] == '1':
            pos.append( (i,line_tuple) )
        else:
            neg.append( (i,line_tuple) )
        
    lines = []
    lines.extend( random.sample(pos,num_seen) )
    lines.extend( random.sample(neg,num_seen) )
    lines = sorted(lines, key = lambda e: e[0])
    for line in lines:
        output_fh.write(''.join(line[1]))
        
        
#def extract_documents(in_fh, out_fh, num_documents):
#    found = 0
#    xml_etree = ET.iterparse(in_fh, events=('end',))
#    
#    is_newsagency = lambda _e: 'type' not in _e.attrib.keys() or _e.attrib['type'] == 'NewsAgency'
#    get_article_id = lambda _e: int(_e.attrib['id'])
#    is_newsagency_pos = lambda _e: is_newsagency(_e) and _e.attrib['relevant'] == 'True'
#    is_newsagency_neg = lambda _e: is_newsagency(_e) and _e.attrib['relevant'] == 'False'
#    
#    for _, element in xml_etree:
#        if element.tag == 'documents': continue
#        
#        found += 1 if is_newsagency(element) else 0
#        article_id = get_article_id(element)
#        
#        article_text = element.text
#        article_headline = re.findall('&lt;headline&gt;.*&lt;/headline&gt;', article_text)
#        article_headline = article_headline[0] if len(article_headline) > 0 else ''
#        article_body = article_text[len(article_headline):]
        
def strip_features(in_fh, out_fh, features):
    """Strips all the documents in *in_fh* to contain only features in *features*
    
    The method reads files in libsvm format.
    
    *features* should be the set of features to retain in the documents.
    """
    
    for line in in_fh:
        doc_class,_,doc_features = line.strip().partition(" ")
        out_fh.write("%s"%doc_class)
        doc_features = dict(re.findall('([0-9]+):([0-9]+)', doc_features))
        keys = set(doc_features.keys())
        keep_features = features & keys 
        for key in keep_features:
            out_fh.write(" %s:%s"%(key, doc_features[key]))
        out_fh.write("\n")

def preprocess_document(doc_text):
    pass
    # should check for duplicates here
        
    # cleanup data
    #    remove http:// addresses
    #    remove &#65533; UTF entities
    #    remove punctuation
    
    # tokenize

    # remove stopwords
    
def compute_feature_counts(input_fh):
    global features
    global posCount
    global negCount
    
    features = collections.defaultdict(lambda: {"score":0,"tp":0,"fp":0,"tn":0,"fn":0})
    posCount = 0
    negCount = 0
#    featuresOrderOfAppearance = []
    document_count = 0
    
    # process articles and count the occurrences of features
    # the non occurrences of features are computed afterwards based on the
    # document count and the occurrences of features
    for document in input_fh:
        document = document.strip()
        if len(document) == 0:
            continue
        
        document_count += 1
        doc_features = document.split(" ")
        
        for feature in doc_features[1:]:
#            individualFeatureSplit = feature.split(":")
            f_token,_,_ = feature.partition(':')
            
            #if the key already exists:
#            if f_token in features:   
                #if article is positive:
            if doc_features[0] == '1':
                features[f_token]['tp'] += 1 #increment tp
            else:
                features[f_token]['fp'] += 1 #increment fp
            
            #if key doesn't exist add it to features dict:
#            else:
#                if doc_features[0] == '1':
#                    features[f_token] = {"score":0,"tp":1,"fp":0,"tn":posCount,"fn":negCount}
#                else:
#                    features[f_token] = {"score":0,"tp":0,"fp":1,"tn":posCount,"fn":negCount}
                
#                featuresOrderOfAppearance.append(individualFeatureSplit[0])
        
        if doc_features[0] == '1':
            posCount += 1
        else:
            negCount += 1
        
    for key in features:
        features[key]["tn"] = (posCount - features[key]["tp"]) #calculate tn
        features[key]["fn"] = (negCount - features[key]["fp"]) #calculate fn
        assert(features[key]['score'] == 0)
        assert(sum(features[key].values()) == document_count)    
    
    return features
