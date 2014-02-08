import logging
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
from discoutils.tokens import DocumentFeature
import os
from thesisgenerator.plugins.thesaurus_loader import Thesaurus
from discoutils.io_utils import write_vectors_to_disk
from thesisgenerator.scripts.build_phrasal_thesauri_offline import do_second_part_without_base_thesaurus, \
    _find_conf_file

__author__ = 'mmb28'
'''
Build a thesaurus of ngrams using observed vectors for the engrams
'''


def do_work(id, svd_dims):
    # SET UP A FEW REQUIRED PATHS
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s ""(line %(lineno)d)\t%(levelname)s : %(""message)s")

    prefix = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit'
    # where are the observed n-gram vectors in tsv format, must be underscore-separated already
    name = 'wiki' if id == 11 else 'gigaw'

    if svd_dims < 0:
        observed_ngram_vectors_file = '%s/observed_vectors/exp%d_AN_NNvectors-cleaned' % (prefix, id)
    else:
        observed_ngram_vectors_file = '%s/exp%d-12b/exp%d-with-obs-phrases-SVD%d.events.filtered.strings' % \
                                      (prefix, id, id, svd_dims)

    # where should the output go
    if svd_dims < 0:
        outdir = '%s/exp%d-13bAN_NN_%s_Observed' % (prefix, id, name)
    else:
        outdir = '%s/exp%d-13bAN_NN_%s-%d_Observed' % (prefix, id, name, svd_dims)

    # where's the byblo conf file
    unigram_thesaurus_dir = '%s/exp%d-13b' % (prefix, id)
    # where's the byblo executable
    byblo_base_dir = '%s/Byblo-2.2.0/' % prefix


    # CREATE BYBLO EVENTS/FEATURES/ENTRIES FILE FROM INPUT
    #  where should these be written
    svd_appendage = '' if svd_dims < 0 else '-SVD%d' % svd_dims
    observed_vector_dir = os.path.dirname(observed_ngram_vectors_file)
    vectors_file = os.path.join(observed_vector_dir, 'exp%d%s.events.filtered.strings' % (id, svd_appendage))
    entries_file = os.path.join(observed_vector_dir, 'exp%d%s.entries.filtered.strings' % (id, svd_appendage))
    features_file = os.path.join(observed_vector_dir, 'exp%d%s.features.filtered.strings' % (id, svd_appendage))

    # do the actual writing
    th = Thesaurus.from_tsv([observed_ngram_vectors_file], aggressive_lowercasing=False)
    mat, cols, rows = th.to_sparse_matrix()
    rows = [DocumentFeature.from_string(x) for x in rows]
    write_vectors_to_disk(mat.tocoo(), rows, cols, vectors_file, features_file, entries_file,
                          entry_filter=lambda feature: feature.type in {'AN', 'NN'})

    logging.info(vectors_file)
    logging.info(entries_file)

    # BUILD A THESAURUS FROM THESE FILES
    os.chdir(byblo_base_dir)
    do_second_part_without_base_thesaurus(_find_conf_file(unigram_thesaurus_dir), outdir,
                                          vectors_file, entries_file, features_file)


if __name__ == '__main__':
    for dims in [30, 300, 1000]: # add -1 to do thesauri without SVD preprocessing of vectors
        do_work(int(sys.argv[1]), dims)
