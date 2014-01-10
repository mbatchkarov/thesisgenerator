import logging
import sys
from thesisgenerator.plugins.tokens import DocumentFeature

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
import os
from thesisgenerator.plugins.thesaurus_loader import Thesaurus
from thesisgenerator.composers.utils import write_vectors_to_disk, julie_transform, reformat_entries
from thesisgenerator.scripts.build_phrasal_thesauri_offline import do_second_part_without_base_thesaurus, \
    _find_conf_file

__author__ = 'mmb28'
'''
Build a thesaurus of ngrams using observed vectors for the engrams
'''


def do_work():
    # SET UP A FEW REQUIRED PATHS
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s ""(line %(lineno)d)\t%(levelname)s : %(""message)s")

    # where are the observed n-gram vectors in tsv format
    observed_ngram_vectors_file = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/Byblo-2.2.0/../' \
                                  'exp10-12-ngrams-observed/exp10_AN_NNvectors' #cleaned.exp10_AN_NNvectors
    # where should the output go
    outdir = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp10-12bAN_NN_gigaw_Observed'
    # where's the byblo conf file
    unigram_thesaurus_dir = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/Byblo-2.2.0/../exp10-12b'
    # where's the byblo executable
    byblo_base_dir = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/Byblo-2.2.0/'

    #  CONVERT THE FILE FROM JULIE'S FORMAT TO MINE
    import re

    pattern = re.compile('(.*):(.*):(.*)')

    def clean(entry):
        a, relation, b = pattern.match(entry).groups()
        if relation == 'amod-HEAD':
            return '{}_{}'.format(a, b)
        elif relation == 'amod-DEP':
            return '{}_{}'.format(b, a)
        elif relation == 'nn-HEAD':
            return '{}_{}'.format(a, b)
        elif relation == 'nn-DEP':
            return '{}_{}'.format(b, a)
        else:
            raise ValueError('Can convert entry %s' % entry)

    observed_ngram_vectors_file = reformat_entries(observed_ngram_vectors_file, '-cleaned', clean)

    # CREATE BYBLO EVENTS/FEATURES/ENTRIES FILE FROM INPUT
    #  where should these be written
    observed_vector_dir = os.path.dirname(observed_ngram_vectors_file)
    vectors_file = os.path.join(observed_vector_dir, 'exp10.events.filtered.strings')
    entries_file = os.path.join(observed_vector_dir, 'exp10.entries.filtered.strings')
    features_file = os.path.join(observed_vector_dir, 'exp10.features.filtered.strings')

    # do the actual writing
    th = Thesaurus.from_tsv([observed_ngram_vectors_file], aggressive_lowercasing=False)
    mat, cols, rows = th.to_sparse_matrix()
    rows = [DocumentFeature.from_string(x) for x in rows]
    write_vectors_to_disk(mat.tocoo(), rows, cols, features_file, entries_file, vectors_file)

    # BUILD A THESAURUS FROM THESE FILES
    os.chdir(byblo_base_dir)
    do_second_part_without_base_thesaurus(_find_conf_file(unigram_thesaurus_dir), outdir,
                                          vectors_file, entries_file, features_file)


if __name__ == '__main__':
    do_work()
