import logging
from discoutils.thesaurus_loader import Vectors
from thesisgenerator.composers.vectorstore import AdditiveComposer
from thesisgenerator.scripts.dump_all_composed_vectors import compose_and_write_vectors
from thesisgenerator.utils.data_utils import get_all_corpora
from thesisgenerator.scripts.extract_NPs_from_labelled_data import get_all_NPs
import numpy as np

"""
Generates a random vector for each NP in all labelled corpora
"""
DIMENSIONALITY = 10 # because why not
out_file = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/random_vectors.gz'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
d = dict()
feats = ['rand%d' % i for i in range(DIMENSIONALITY)]
for phrase in get_all_NPs():
    vector = np.random.random(DIMENSIONALITY)
    d[phrase.tokens_as_str()] = zip(feats, vector)

v = Vectors(d)
v.to_tsv(out_file,  gzipped=True)