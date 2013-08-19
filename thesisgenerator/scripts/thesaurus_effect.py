# coding=utf-8
"""
Used to investigate the effect a thesaurus has on classification performance.
Given a run log and PostVectorizerDump.csv for that run (which contains the
set of training feature vectors) one can look at the "association" between a
feature and a class (quite basic, needs a more sophisticated approach) and
inspect what features are being inserted. For instance,
one might find features strongly associated with class 6 are inserted into a
test document of class 4, which is bad.

Only appli
"""

# TODO compare 22-5 (baseline) vs 26-5 and 30-5 (gigaw/wiki respectively)
import logging
import cPickle as pickle

from pandas.io.parsers import read_csv
from numpy import *
import numpy.testing as t



# th_vectors_file = 'PostVectDump_exp{}-{}_tr0.csv'.format(with_th, subexp)
# th_log_file = 'conf/exp{0}/logs/exp{0}-{1}.log'.format(with_th, subexp)


def get_classifier(exp_no, subexp):
    logging.info('Loading pipeline from exp {}-{}'.format(exp_no, subexp))
    return pickle.load(open('exp{}-{}-pipeline.pickle'.format(exp_no, subexp)))


def get_data(exp_no, subexp):
    logging.info('Loading data from exp {}-{}'.format(exp_no, subexp))
    vectors_file = 'PostVectDump_exp{}-{}_tr1.csv'.format(exp_no, subexp)

    # find feature weights and document targets from feature vector file
    df = read_csv(vectors_file)
    first_feature_column = list(df.columns).index('nonzero_feats') + 1

    features = df.ix[:, first_feature_column:].as_matrix()
    targets = array(df['target'])
    return features, targets


baseline, with_th = 22, 26
subexp = 5
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %(message)s",
                    datefmt='%d.%m.%Y %H:%M:%S')

clf = get_classifier(22, 5)
features1, targets1 = get_data(baseline, subexp)
features2, targets2 = get_data(with_th, subexp)
# targets need to be the same because the same data point are used for training
t.assert_allclose(targets1, targets2)
# we expect the features to be different because each thesaurus will intervene differently
t.assert_allclose(features1, features2)





# log_file = 'conf/exp{0}/logs/exp{0}-{1}.log'.format(exp_no, subexp)
# # what class a feature most contributes to
# classes = dict(df.groupby('target').sum().idxmax()[1:])
# # what class a document belongs to
# targets = dict(df['target'])
#
# #read log file
# txt = [x.strip() for x in open(log_file).readlines()]
# import re
#
# replacements = re.findall(
#     r'Replacement. Doc (\d+): .*/.* --> (.*/.*), sim',
#     '\n'.join(txt)
# )
#
# correct_labels = []
# inserted_labels = []
# for (doc, newtok) in replacements:
#     try:
#         a = targets[int(doc)]
#         b = classes[newtok]
#         correct_labels.append(a)
#         inserted_labels.append(b)
#     except KeyError:
#         pass
#
# print 'Incorrect insertion rate is ', count_nonzero(
#     array(inserted_labels) - array(correct_labels)) / float(
#     len(inserted_labels))
