import argparse
import logging
import re
import sys
from discoutils.reduce_dimensionality import filter_out_infrequent_entries

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
import os
from discoutils.tokens import DocumentFeature
from discoutils.thesaurus_loader import Thesaurus, Vectors
from discoutils.io_utils import write_vectors_to_disk
import thesisgenerator.scripts.build_phrasal_thesauri_offline as offline
from thesisgenerator.utils.misc import noop
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


    # where's the byblo conf file
    unigram_thesaurus_dir = '%s/exp%d-%db' % (prefix, corpus, features)
    # where's the byblo executable
    byblo_base_dir = '%s/Byblo-2.2.0/' % prefix

    if svd_dims == 0:
        observed_ngram_vectors_file = '%s/observed_vectors/exp%d-%d_AN_NNvectors-cleaned' % (prefix, corpus, features)
        unigram_thesaurus_dir = os.path.abspath(os.path.join(byblo_base_dir, '..',
                                                             'exp%d-%db' % (corpus, features)))
        observed_unigram_vectors_file = offline._find_events_file(unigram_thesaurus_dir)
    else:
        # contain SVD-reduced N,J and NP observed vectors
        observed_ngram_vectors_file = '%s/exp%d-%db/exp%d-with-obs-phrases-SVD%d.events.filtered.strings' % \
                                      (prefix, corpus, features, corpus, svd_dims)

    logging.info('Using observed events file %s ', observed_ngram_vectors_file)
    # where should the output go
    if svd_dims == 0:
        outdir = '%s/exp%d-%dbAN_NN_%s_Observed' % (prefix, corpus, features, name)
    else:
        outdir = '%s/exp%d-%dbAN_NN_%s-%d_Observed' % (prefix, corpus, features, name, svd_dims)

    # CREATE BYBLO EVENTS/FEATURES/ENTRIES FILE FROM INPUT
    # where should these be written
    svd_appendage = '' if svd_dims == 0 else '-SVD%d' % svd_dims
    observed_vector_dir = os.path.dirname(observed_ngram_vectors_file)
    vectors_file = os.path.join(observed_vector_dir, 'exp%d%s.events.filtered.strings' % (corpus, svd_appendage))
    entries_file = os.path.join(observed_vector_dir, 'exp%d%s.entries.filtered.strings' % (corpus, svd_appendage))
    features_file = os.path.join(observed_vector_dir, 'exp%d%s.features.filtered.strings' % (corpus, svd_appendage))

    # do the actual writing
    if svd_dims:
        th = Vectors.from_tsv(observed_ngram_vectors_file, lowercasing=False)
    else:
        # th = Vectors.from_tsv([observed_ngram_vectors_file, observed_unigram_vectors_file])
        # read and merge unigram and n-gram vectors files
        th0 = Vectors.from_tsv(observed_unigram_vectors_file)
        th1 = Vectors.from_tsv(observed_ngram_vectors_file)
        data0 = th0._obj
        data0.update(th1._obj)
        th = Vectors(data0)
        del th0, th1

    desired_counts_per_feature_type = [('N', 20000), ('V', 0), ('J', 10000), ('RB', 0), ('AN', 1e10), ('NN', 1e10)]
    mat, pos_tags, rows, cols = filter_out_infrequent_entries(desired_counts_per_feature_type, th)
    # mat, cols, rows = th.to_sparse_matrix() # if there are only NPs do this
    # rows = [DocumentFeature.from_string(x) for x in rows]
    write_vectors_to_disk(mat.tocoo(), rows, cols, vectors_file, features_file, entries_file,
                          entry_filter=lambda feature: feature.type in {'AN', 'NN', '1-GRAM'},
                          gzipped=True)

    logging.info(vectors_file)
    logging.info(entries_file)

    # BUILD A THESAURUS FROM THESE FILES
    os.chdir(byblo_base_dir)
    offline.do_second_part_without_base_thesaurus(offline._find_conf_file(unigram_thesaurus_dir), outdir,
                                                  vectors_file, entries_file, features_file)


def do_work_socher(baronify):
    """
    Formats the files output by Socher (2011)'s matlab code into byblo-compatible files.

    Before running this a list of all phrases needs to be extracted from the labelled data, and these need to
    be composed with Socher's matlab code. See note "Socher vectors" in Evernote.

    :param baronify:
    """
    # SET UP A FEW REQUIRED PATHS
    # where are the composed n-gram vectors, must contain parsed.txt, phrases.txt and outVectors.txt
    # before running this function, put all phrases to be composed in parsed.txt, wrapping them
    # up to make them look like fragment of a syntactic parser. Do NOT let the Stanford parser that ships with
    # that code run.
    # must also contain 'socher.bybloconf.txt' and a 'thesaurus' subdirectory
    prefix = '/mnt/lustre/scratch/inf/mmb28/FeatureExtractionToolkit'

    socher_base_dir = os.path.join(prefix, 'socher_vectors')  # copy downloaded content here
    socher_phrases_file = os.path.join(socher_base_dir, 'phrases.txt')
    socher_parsed_file = os.path.join(socher_base_dir, 'parsed.txt')
    socher_vectors_file = os.path.join(socher_base_dir, 'outVectors.txt')
    byblo_conf_file = os.path.join(socher_base_dir, 'socher.bybloconf.txt')
    output_dir = os.path.join(socher_base_dir, 'thesaurus')
    if baronify:
        output_dir = os.path.join(socher_base_dir, 'thesaurus_baronified')
    logging.info('Using phrases events file %s', socher_vectors_file)

    # where's the byblo executable
    byblo_base_dir = '%s/Byblo-2.2.0/' % prefix

    an_regex = re.compile("\(NP \(JJ (\S+)\) \(NN (\S+)\)\)\)")
    nn_regex = re.compile("\(NP \(NN (\S+)\) \(NN (\S+)\)\)\)")
    unigram_regex = re.compile("\(NP \((NN|JJ) (\S+)\)\)\)")

    # get a list of all phrases that we attempted to compose
    composed_phrases = []
    with open(socher_parsed_file) as infile:
        for line in infile:
            # check if this is an AN
            matches = an_regex.findall(line)
            if matches:
                d = DocumentFeature.from_string('{}/J_{}/N'.format(*matches[0]))
                composed_phrases.append(d)
                continue

            # check if this is an NN
            matches = nn_regex.findall(line)
            if matches:
                d = DocumentFeature.from_string('{}/N_{}/N'.format(*matches[0]))
                composed_phrases.append(d)
                continue

            # check if this is a unigram
            matches = unigram_regex.findall(line)
            if matches:
                d = DocumentFeature.from_string('{1}/{0}'.format(*matches[0])[:-1])
                composed_phrases.append(d)
                continue

            # if we got to here, something is amiss
            # this line is neither empty nor parser bolerplate- usually poorly stripped HTML
            # pretend nothing is wrong, the composer would not have dealt with this, so
            # this phrase will get removed by the filter below
            if '(ROOT' not in line and len(line.strip()) > 0:
                composed_phrases.append(None)  # indicates that something is wrong

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
    vectors_file = os.path.join(output_dir, 'socher.events.filtered.strings')
    entries_file = os.path.join(output_dir, 'socher.entries.filtered.strings')
    features_file = os.path.join(output_dir, 'socher.features.filtered.strings')

    # do the actual writing
    write_vectors_to_disk(
        sp.coo_matrix(mat),
        composed_phrases,
        ['RAE-feat%d' % i for i in range(100)],  # Socher provides 100-dimensional vectors
        vectors_file,
        gzipped=True
    )

    # if baronify:
    # entries_file, features_file, vectors_file = baronify_files(entries_file, features_file, vectors_file)


    # this function writes its output to the correct location, there is no need to run a monkey-patched version
    # of the function below to symlink output
    # BUILD A THESAURUS FROM THESE FILES
    # os.chdir(byblo_base_dir)
    # offline.do_second_part_without_base_thesaurus(byblo_conf_file, output_dir,
    #                                               vectors_file, entries_file, features_file)


def get_cmd_parser():
    from thesisgenerator.scripts.build_phrasal_thesauri_offline import get_corpus_features_cmd_parser

    parser = argparse.ArgumentParser(parents=[get_corpus_features_cmd_parser()])
    # add options specific to this script here
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--svd', choices=(0, 30, 100, 300, 1000), nargs='+', type=int,
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

    offline.run_byblo = noop
    offline.reindex_all_byblo_vectors = noop

    if parameters.socher:
        do_work_socher(parameters.baronify)
    else:
        corpus = 10 if parameters.corpus == 'gigaword' else 11
        features = 12 if parameters.features == 'dependencies' else 13
        for dims in parameters.svd:
            do_work(corpus, features, dims)
