import argparse
import logging
import re
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
import os
from discoutils.tokens import DocumentFeature
from discoutils.thesaurus_loader import Thesaurus
from discoutils.io_utils import write_vectors_to_disk
from thesisgenerator.scripts.build_phrasal_thesauri_offline import do_second_part_without_base_thesaurus, \
    _find_conf_file
import numpy as np
import scipy.sparse as sp
from operator import itemgetter


'''
Build a thesaurus of ngrams using observed vectors for the engrams
'''


def do_work(corpus, features, svd_dims):
    # SET UP A FEW REQUIRED PATHS

    prefix = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit'
    # where are the observed n-gram vectors in tsv format, must be underscore-separated already
    name = 'wiki' if corpus == 11 else 'gigaw'

    if svd_dims == 0:
        observed_ngram_vectors_file = '%s/observed_vectors/exp%d-%d_AN_NNvectors-cleaned' % (prefix, corpus, features)
    else:
        observed_ngram_vectors_file = '%s/exp%d-%db/exp%d-with-obs-phrases-SVD%d.events.filtered.strings' % \
                                      (prefix, corpus, features, corpus, svd_dims)

    logging.info('Using observed events file %s', observed_ngram_vectors_file)
    # where should the output go
    if svd_dims == 0:
        outdir = '%s/exp%d-%dbAN_NN_%s_Observed' % (prefix, corpus, features, name)
    else:
        outdir = '%s/exp%d-%dbAN_NN_%s-%d_Observed' % (prefix, corpus, features, name, svd_dims)

    # where's the byblo conf file
    unigram_thesaurus_dir = '%s/exp%d-%db' % (prefix, corpus, features)
    # where's the byblo executable
    byblo_base_dir = '%s/Byblo-2.2.0/' % prefix


    # CREATE BYBLO EVENTS/FEATURES/ENTRIES FILE FROM INPUT
    #  where should these be written
    svd_appendage = '' if svd_dims == 0 else '-SVD%d' % svd_dims
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


def do_work_socher():
    # SET UP A FEW REQUIRED PATHS
    # where are the composed n-gram vectors, must contain parsed.txt, phrases.txt and outVectors.txt
    # before running this function, put all phrases to be composed in parsed.txt, wrapping them
    # up to make them look like fragment of a syntactic parser. Do NOT let the Stanford parser that ships with
    # that code run.
    prefix = '/mnt/lustre/scratch/inf/mmb28/FeatureExtractionToolkit'
    # where should the output go
    outdir = os.path.join(prefix, 'exp10-12bAN_NN_gigaw_Socher')

    socher_base_dir = '/Volumes/Storage/Downloads/codeRAEVectorsNIPS2011'
    socher_phrases_file = os.path.join(socher_base_dir, 'phrases.txt')
    socher_parsed_file = os.path.join(socher_base_dir, 'parsed.txt')
    socher_vectors_file = os.path.join(socher_base_dir, 'outVectors.txt')
    formatted_vectors_dir = os.path.join(prefix, 'socher_vectors')

    logging.info('Using phrases events file %s', socher_vectors_file)


    # where's the byblo conf file
    unigram_thesaurus_dir = '%s/exp%d-%db' % (prefix, corpus, features)
    # where's the byblo executable
    byblo_base_dir = '%s/Byblo-2.2.0/' % prefix

    an_regex = re.compile("\(NP \(JJ (\S+)\) \(NN (\S+)\)\)\)")
    nn_regex = re.compile("\(NP \(NN (\S+)\) \(NN (\S+)\)\)\)")

    # get a list of all phrases that we attempted to compose
    composed_phrases = []
    with open(socher_parsed_file) as infile:
        for line in infile:
            # check if this is an AN
            matches = an_regex.findall(line)
            if matches:
                d = DocumentFeature.from_string('{}/J_{}/N'.format(*matches[0]))
                composed_phrases.append(d)
            else:
                # check if this is an NN
                matches = nn_regex.findall(line)
                if matches:
                    d = DocumentFeature.from_string('{}/N_{}/N'.format(*matches[0]))
                    composed_phrases.append(d)

    # get a list of all phrases where composition worked (no unknown words)
    with open(socher_phrases_file) as infile:
        success = [i for i, line in enumerate(infile) if '*UNKNOWN*' not in line]
    # pick out just the phrases that composes successfully
    composed_phrases = itemgetter(*success)(composed_phrases)

    # load all vectors, remove these containing unknown words
    mat = np.loadtxt(socher_vectors_file, delimiter=',')
    mat = mat[success, :]
    assert len(composed_phrases) == mat.shape[0]  # same number of rows

    # CREATE BYBLO EVENTS/FEATURES/ENTRIES FILE FROM INPUT
    # Pretend the file name is 10-12, i.e. these vectors came from giga deps (for consistency with other exps)
    vectors_file = os.path.join(formatted_vectors_dir, 'exp10-12.events.filtered.strings')
    entries_file = os.path.join(formatted_vectors_dir, 'exp10-12.entries.filtered.strings')
    features_file = os.path.join(formatted_vectors_dir, 'exp10-12.features.filtered.strings')

    logging.info(vectors_file)
    logging.info(entries_file)
    # do the actual writing
    write_vectors_to_disk(
        sp.coo_matrix(mat),
        composed_phrases,
        ['RAE-feat%d' % i for i in range(100)],  # Socher provides 100-dimensional vectors
        vectors_file,
        features_file,
        entries_file
    )


    # BUILD A THESAURUS FROM THESE FILES
    os.chdir(byblo_base_dir)
    do_second_part_without_base_thesaurus(_find_conf_file(unigram_thesaurus_dir), outdir,
                                          vectors_file, entries_file, features_file)


def get_cmd_parser():
    from thesisgenerator.scripts.build_phrasal_thesauri_offline import get_corpus_features_cmd_parser

    parser = argparse.ArgumentParser(parents=[get_corpus_features_cmd_parser()])
    # add options specific to this script here
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--svd', choices=(0, 30, 300, 1000), nargs='+', type=int,
                       help='What SVD dimensionalities to build observed-vector thesauri from. '
                            'Vectors must have been produced and reduced already. 0 stand for unreduced.')
    group.add_argument('--socher', action='store_true',
                       help='If set, Socher pre-composed RAE vectors will be used.')
    return parser


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %(message)s")

    parameters = get_cmd_parser().parse_args()
    logging.info(parameters)

    if parameters.socher:
        do_work_socher()
    else:
        corpus = 10 if parameters.corpus == 'gigaword' else 11
        features = 12 if parameters.features == 'dependencies' else 13
        for dims in parameters.svd:
            do_work(corpus, features, dims)
