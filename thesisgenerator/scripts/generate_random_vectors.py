import sys
sys.path.append('.')
import logging
from discoutils.thesaurus_loader import Vectors
from thesisgenerator.scripts.extract_NPs_from_labelled_data import get_all_NPs
import numpy as np

"""
Generates a random vector for each NP in all labelled corpora
"""
DIMENSIONALITY = 10 # because why not
out_file = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/random_vectors.gz'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
np.random.seed(0)
d = dict()
feats = ['rand%d' % i for i in range(DIMENSIONALITY)]
for phrase in get_all_NPs():
    vector = np.random.random(DIMENSIONALITY)
    d[phrase.tokens_as_str()] = zip(feats, vector)

v = Vectors(d)
v.to_tsv(out_file,  gzipped=True)