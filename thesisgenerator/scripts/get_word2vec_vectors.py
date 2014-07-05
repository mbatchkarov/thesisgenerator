# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>
import os, sys
# os.chdir('/mnt/lustre/scratch/inf/mmb28/thesisgenerator')
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

import gensim, logging, errno

from discoutils.thesaurus_loader import Thesaurus
import numpy as np
from thesisgenerator.plugins.tokenizers import XmlTokenizer

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# <codecell>

# inputs
conll_data_dir = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/data/gigaword-afe-split-tagged-parsed/gigaword/'
pos_only_data_dir = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/data/gigaword-afe-split-pos/gigaword/'
# outputs
unigram_events_file = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/word2vec_vectors/thesaurus/word2vec.events.filtered.strings'
unigram_thes_file = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/word2vec_vectors/word2vec.unigram.thesaurus.txt'

# operation parameters
MIN_COUNT = 50
WORKERS = 4
pos_map = XmlTokenizer.pos_coarsification_map

# <markdowncell>

# Data formatting
# =========
# `word2vec` produces vectors for words, such as `computer`, whereas the rest of my experiments assume there are augmented with a PoS tag, e.g. `computer/N`. To get around that, start with a directory of conll-formatted files such as 
# ```
# 1	Anarchism	Anarchism	NNP	MISC	5	nsubj
# 2	is	be	VBZ	O	5	cop
# 3	a	a	DT	O	5	det
# 4	political	political	JJ	O	5	amod
# 5	philosophy	philosophy	NN	O	0	root
# ```
# 
# and convert them to pos-augmented format (using coarse tags like Petrov's):
# 
# ```
# Anarchism/N is/V a/DET ....
# ```

# <codecell>

try:
    os.makedirs(pos_only_data_dir)
except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(pos_only_data_dir):
        pass
    else: raise

for filename in os.listdir(conll_data_dir):
    outfile_name = os.path.join(pos_only_data_dir, filename)
    logging.info('Reformatting %s to %s', filename, outfile_name)
    with open(os.path.join(conll_data_dir, filename)) as infile, open(outfile_name, 'w') as outfile:
        for line in infile:
            if not line.strip(): # conll empty line = sentence boundary
                outfile.write('.\n')
                continue
            idx, word, lemma, pos, ner, dep, _ = line.strip().split('\t')
            outfile.write('%s/%s '%(lemma.lower(), pos_map[pos]))

# <codecell>

# train a word2vec model
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()


logging.info('Training word2vec')
sentences = MySentences(pos_only_data_dir)  # a memory-friendly iterator
model = gensim.models.Word2Vec(sentences, workers=WORKERS, min_count=MIN_COUNT)

# <codecell>

# sanity check
logging.info(model.most_similar('computer/N', topn=20))
logging.info(model['computer/N'].shape)
logging.info(model.similarity('computer/N', 'software/N'))

# <codecell>

# get word2vec vectors for each word, write to TSV
vectors = dict()
dimension_names = ['word2vec_feat%02d'%i for i in range(100)] # word2vec produces 100-dim vectors
for word in model.vocab.keys():
    vectors[word] = zip(dimension_names, model[word])
th1 = Thesaurus(vectors)
th1.to_tsv(unigram_events_file, preserve_order=True)

# <codecell>
if False:
    # build a thesaurus out of the nearest neighbours of each unigram and save it to TSV
    # this is a little incompatible with the rest of my thesauri as it uses the first PoS-augmented form for each word2vec word is used
    # nevertheless, it's useful to compare these neighbours to Byblo's neighbours as a sanity check
    mythes = dict()
    for word in model.vocab.keys():
        # if word == 'computer/N':
        #     print model.most_similar(word, topn=10)
        mythes[word] = model.most_similar(word, topn=10)
    # print len(mythes), np.mean([len(foo) for foo in mythes.values()])
    Thesaurus(mythes).to_tsv(unigram_thes_file, preserve_order=True)

# <codecell>

# print model['computer/N'][:60]

