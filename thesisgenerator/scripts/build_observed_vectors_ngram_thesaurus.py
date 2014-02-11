import logging
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
import os
from discoutils.tokens import DocumentFeature
from discoutils.thesaurus_loader import Thesaurus
from discoutils.io_utils import write_vectors_to_disk
from thesisgenerator.scripts.build_phrasal_thesauri_offline import do_second_part_without_base_thesaurus, \
    _find_conf_file, read_configuration

__author__ = 'mmb28'
'''
Build a thesaurus of ngrams using observed vectors for the engrams
'''


def do_work(corpus, features, svd_dims):
    # SET UP A FEW REQUIRED PATHS

    prefix = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit'
    # where are the observed n-gram vectors in tsv format, must be underscore-separated already
    name = 'wiki' if corpus == 11 else 'gigaw'

    if svd_dims < 0:
        observed_ngram_vectors_file = '%s/observed_vectors/exp%d-%d_AN_NNvectors-cleaned' % (prefix, corpus, features)
    else:
        observed_ngram_vectors_file = '%s/exp%d-%db/exp%d-with-obs-phrases-SVD%d.events.filtered.strings' % \
                                      (prefix, corpus, features, corpus, svd_dims)

    logging.info('Using observed events file %s', observed_ngram_vectors_file)
    # where should the output go
    if svd_dims < 0:
        outdir = '%s/exp%d-%dbAN_NN_%s_Observed' % (prefix, corpus, features, name)
    else:
        outdir = '%s/exp%d-%dbAN_NN_%s-%d_Observed' % (prefix, corpus, features, name, svd_dims)

    # where's the byblo conf file
    unigram_thesaurus_dir = '%s/exp%d-%db' % (prefix, corpus, features)
    # where's the byblo executable
    byblo_base_dir = '%s/Byblo-2.2.0/' % prefix


    # CREATE BYBLO EVENTS/FEATURES/ENTRIES FILE FROM INPUT
    #  where should these be written
    svd_appendage = '' if svd_dims < 0 else '-SVD%d' % svd_dims
    observed_vector_dir = os.path.dirname(observed_ngram_vectors_file)
    vectors_file = os.path.join(observed_vector_dir, 'exp%d%s.events.filtered.strings' % (corpus, svd_appendage))
    entries_file = os.path.join(observed_vector_dir, 'exp%d%s.entries.filtered.strings' % (corpus, svd_appendage))
    features_file = os.path.join(observed_vector_dir, 'exp%d%s.features.filtered.strings' % (corpus, svd_appendage))

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
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %(message)s")

    parameters = read_configuration()
    logging.info(parameters)

    corpus = 10 if parameters.corpus == 'gigaword' else 11
    features = 12 if parameters.features == 'dependencies' else 13

    for dims in [30, 300, 1000]:  # todo add -1 to do thesauri without SVD preprocessing of vectors
        do_work(corpus, features, dims)
