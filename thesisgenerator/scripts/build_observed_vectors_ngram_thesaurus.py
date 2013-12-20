import logging
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
import os
from thesisgenerator.plugins.thesaurus_loader import Thesaurus
from thesisgenerator.composers.utils import write_vectors_to_disk, julie_transform
from thesisgenerator.plugins.tokenizers import DocumentFeature
from thesisgenerator.scripts.build_phrasal_thesauri_offline import do_second_part2

__author__ = 'mmb28'
'''
Build a thesaurus of ngrams using observed vectors for the engrams
'''

def do_work():
    # SET UP A FEW REQUIRED PATHS
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s ""(line %(lineno)d)\t%(levelname)s : %(""message)s")

    byblo_base_dir = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/Byblo-2.2.0/' # trailing slash required
    all_ngram_vectors_dir = os.path.join(byblo_base_dir, '..', 'exp10-12-ngrams')
    unigram_thesaurus_dir = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/Byblo-2.2.0/../exp10-12'
    observed_ngram_vectors_dir = all_ngram_vectors_dir + '-observed'
    # where are the observed vectors?
    observed_vector_file = os.path.join(observed_ngram_vectors_dir, 'cleaned.exp10_AN_NNvectors')


    vectors_file = os.path.join(observed_ngram_vectors_dir, 'exp10.events.filtered.strings')
    entries_file = os.path.join(observed_ngram_vectors_dir, 'exp10.entries.filtered.strings')
    features_file = os.path.join(observed_ngram_vectors_dir, 'exp10.features.filtered.strings')

    th = Thesaurus([observed_vector_file], aggressive_lowercasing=False)
    mat, cols, rows = th.to_sparse_matrix()
    rows = [DocumentFeature.from_string(x) for x in rows]

    os.chdir(byblo_base_dir)
    write_vectors_to_disk(mat.tocoo(), rows, cols, features_file, entries_file, vectors_file)
    outdir = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp10-12AN_NN_gigaw_Observed'
    do_second_part2(unigram_thesaurus_dir, vectors_file, entries_file, features_file, copy_to_dir=outdir)


if __name__ == '__main__':
    do_work()
