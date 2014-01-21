#!/bin/bash
# convers Julie's observed vectors to my format

cd /mnt/lustre/scratch/inf/mmb28/thesisgenerator

# put all files from the same source corpus together
x=/mnt/lustre/scratch/inf/juliewe/Compounds/data/miro/observed_vectors/
y=/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/observed_vectors/
cat $x/*wiki* > $y/exp11_AN_NNvectors
cat $x/*giga* > $y/exp10_AN_NNvectors

# convert to underscore-separated
python -c "
import re; 
from thesisgenerator.scripts.build_observed_vectors_ngram_thesaurus import clean; 
from thesisgenerator.composers.utils import write_vectors_to_disk, julie_transform, reformat_entries; 
for i in [10,11]:
    observed_ngram_vectors_file = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/observed_vectors/exp%d_AN_NNvectors' % i; 
    reformat_entries(observed_ngram_vectors_file, '-cleaned', clean);
"
