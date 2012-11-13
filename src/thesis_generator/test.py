'''
Created on Oct 24, 2012

@author: ml249
'''

import sys
import gzip
import cPickle as pickle
import timeit

def strip_features_re(in_fh, out_fh, features):
    """Strips all the documents in *in_fh* to contain only features in *features*
    
    The method reads files in libsvm format.
    
    *features* should be the set of features to retain in the documents.
    """
    import re
    for line in in_fh:
        doc_class,_,doc_features = line.strip().partition(" ")
        out_fh.write("%s"%doc_class)
        doc_features = dict(re.findall('([0-9]+):([0-9]+)', doc_features))
        keys = set(doc_features.keys())
        for key in features & keys:
            out_fh.write(" %s:%s"%(key, doc_features[key]))
        out_fh.write("\n")

def strip_features_parser(in_fh, out_fh, features):
    """Strips all the documents in *in_fh* to contain only features in *features*
    
    The method reads files in libsvm format.
    
    *features* should be the set of features to retain in the documents.
    """
    nums = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}

    def parse_class(fh, buf=[]):
        while buf[-1] != ' ':
            buf.append(fh.read(1))
        return buf
    
    def parse_feature_tuples(fh,buf):
        b = ''
        while buf[-1] != '\n' and b != '\n':
            if parse_feature_key(fh, buf):
                buf = parse_feature_value(fh, buf)
            else:
                b = fh.read(1)
                while b != ' ' and b != '\n':
                    b = fh.read(1)
#            yield (key,value)
#        raise StopIteration
        return buf
        
    def parse_feature_key(fh,buf=[]):
        local = []
        local.append(fh.read(1))
        while local[-1] in nums:
            local.append(fh.read(1))

        if ''.join(local[:-1]) in features:
            buf.extend(local)
            return True
        else:
            return False
        
    def parse_feature_value(fh,buf=[]):
        buf.append(fh.read(1))
        while buf[-1] in nums:
            buf.append(fh.read(1))
        return buf
    
    try:
        b = in_fh.read(1)
        buf = [b]
        while b != '':
            buf = parse_class(in_fh, buf)
#            print buf
#            out_fh.write(''.join(buf))
            buf = parse_feature_tuples(in_fh, buf)
            
#            for key,value in :
#                if key in features:
#                    out_fh.write(" %s:%s"%(key, value))
            
            out_fh.write(''.join(buf))
            out_fh.write("\n")
            b = in_fh.read(1)
            buf = [b]
    except EOFError: pass

def do_test(func, in_fname, out_fname):
    with open(in_fname, 'r') as in_fh, open(out_fname, 'w') as out_fh: 
        features = pickle.load(open('/usr/local/scratch/TEMP/features.pickle','r'))
        getattr(sys.modules[__name__], func)(in_fh, out_fh, features)

if __name__ == "__main__":
#    in_fh = open("/usr/local/scratch/TEMP/train/kazakhmys.train", 'rb')
#    out_fh = open("/usr/local/scratch/TEMP/parser.train.txt", 'w')
#    features = pickle.load(open('/usr/local/scratch/TEMP/features.pickle','r'))
#    strip_features_parser(in_fh, out_fh,features)
    
    print 'RegExp', timeit.timeit(stmt='do_test("strip_features_re", "/usr/local/scratch/TEMP/train/kazakhmys.train", "/usr/local/scratch/TEMP/re.train.txt")', setup='from __main__ import do_test', number=100) / 100.0
    print 'Parser', timeit.timeit(stmt='do_test("strip_features_parser", "/usr/local/scratch/TEMP/train/kazakhmys.train", "/usr/local/scratch/TEMP/parser.train.txt")', setup='from __main__ import do_test', number=100) / 100.0
