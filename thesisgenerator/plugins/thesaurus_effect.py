"""
Used to investigate the effect a thesaurus has on classification performance.
Given a run log and PostVectorizerDump.csv for that run (which contains the
set of training feature vectors) one can look at the "association" between a
feature and a class (quite basic, needs a more sophisticated approach) and
inspect what features are being inserted. For instance,
one might find features strongly associated with class 6 are inserted into a
test document of class 4, which is bad.
"""

from pandas.io.parsers import read_csv
from numpy import *

FEATURE_VECTORS_FILE = 'PostVectorizerDump.csv'
LOG_FILE = 'conf/exp6/exp6-1.log'

# find feature weights and document targets from feature vector file
df = read_csv(FEATURE_VECTORS_FILE)
# what class a feature most contributes to
classes = dict(df.groupby('target').sum().idxmax()[1:])
# what class a document belongs to
targets = dict(df['target'])

#read log file
txt = [x.strip() for x in open(LOG_FILE).readlines()]
import re

replacements = re.findall(
    r'Replacement. Doc (\d+): .*/.* --> (.*/.*), sim',
    '\n'.join(txt)
)

correct_labels = []
inserted_labels = []
for (doc, newtok) in replacements:
    try:
        a = targets[int(doc)]
        b = classes[newtok]
        correct_labels.append(a)
        inserted_labels.append(b)
    except KeyError:
        pass

print 'Incorrect insertion rate is ', count_nonzero(
    array(inserted_labels) - array(correct_labels)) / float(
    len(inserted_labels))