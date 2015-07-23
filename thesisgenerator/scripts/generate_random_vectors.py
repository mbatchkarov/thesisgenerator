import sys

sys.path.append('.')
import logging
from discoutils.thesaurus_loader import DenseVectors
from thesisgenerator.scripts.extract_NPs_from_labelled_data import get_all_NPs_VPs
import numpy as np
import pandas as pd

"""
Generates a random vector for each NP in all labelled corpora
"""
DIMENSIONALITY = 100
out_file = '/lustre/scratch/inf/mmb28/FeatureExtractionToolkit/random_vectors.gz'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
np.random.seed(0)
feats = ['rand%d' % i for i in range(DIMENSIONALITY)]
phrases = list(get_all_NPs_VPs(include_unigrams=True))
vectors = np.random.random((len(phrases), DIMENSIONALITY))

v = DenseVectors(pd.DataFrame(vectors, index=phrases, columns=feats))
v.to_tsv(out_file, dense_hd5=True)