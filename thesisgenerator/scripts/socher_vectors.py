import sys
sys.path.append('.')
import argparse
import logging
from operator import itemgetter
import os
import re
import numpy as np
import scipy.sparse as sp
from discoutils.tokens import DocumentFeature
from discoutils.io_utils import write_vectors_to_disk
from discoutils.cmd_utils import run_and_log_output
from discoutils.misc import temp_chdir, mkdirs_if_not_exists
from thesisgenerator.composers.vectorstore import (MultiplicativeComposer, AdditiveComposer,
                                                   RightmostWordComposer, LeftmostWordComposer,
                                                   VerbComposer, compose_and_write_vectors)
from thesisgenerator.utils.misc import force_symlink

# SET UP A FEW REQUIRED PATHS
# where are the composed n-gram vectors, must contain parsed.txt, phrases.txt and outVectors.txt
# before running this function, put all phrases to be composed in parsed.txt, wrapping them
# up to make them look like fragment of a syntactic parser. Do NOT let the Stanford parser that ships with
# that code run.
prefix = '/mnt/lustre/scratch/inf/mmb28/FeatureExtractionToolkit'
socher_base_dir = os.path.join(prefix, 'socher_vectors')  # copy downloaded content here

# the two paths below needs to point to the same thing
phrases_to_compose = os.path.join(prefix, '..', 'thesisgenerator',
                                  'features_in_labelled', 'socher.txt')
socher_input_file = os.path.join(socher_base_dir, 'parsed.txt')
plaintext_socher_input_file = os.path.join(prefix, '..', 'thesisgenerator',
                                  'features_in_labelled', 'all_features.txt')

socher_output_phrases_file = os.path.join(socher_base_dir, 'phrases.txt')
socher_output_vectors_file = os.path.join(socher_base_dir, 'outVectors.txt')

# output of reformat stage
output_dir = os.path.join(socher_base_dir, 'composed')
mkdirs_if_not_exists(output_dir)
vectors_file = os.path.join(output_dir, 'AN_NN_turian_Socher.events.filtered.strings')


def run_socher_code():
    # symlink the file Socher's code expects to where the list of phrases I'm interested is
    force_symlink(phrases_to_compose, socher_input_file)
    with temp_chdir(socher_base_dir):
        run_and_log_output('./phrase2Vector.sh')  # this takes a while
        # output files are phrases.txt and outVectors.txt


def reformat_socher_vectors():
    """
    Formats the files output by Socher (2011)'s matlab code into byblo-compatible files.

    Before running this a list of all phrases needs to be extracted from the labelled data, and these need to
    be composed with Socher's matlab code. See note "Socher vectors" in Evernote.

    """
    logging.info('Reformatting events file %s ---> %s',
                 socher_output_vectors_file, vectors_file)

    # socher's code removes all PoS tags, so we can't translate his output
    # back to a DocumentFeature. Let's read the input to his code instead and
    # get the corresponding output vectors
    # get a list of all phrases that we attempted to compose
    with open(plaintext_socher_input_file) as infile:
        composed_phrases = [DocumentFeature.from_string(line.strip()) for line in infile]

    # get a list of all phrases where composition worked (no unknown words)
    with open(socher_output_phrases_file) as infile:
        success = [i for i, line in enumerate(infile) if '*UNKNOWN*' not in line]
        # pick out just the phrases that composes successfully
    composed_phrases = itemgetter(*success)(composed_phrases)

    # load all vectors, remove these containing unknown words
    mat = np.loadtxt(socher_output_vectors_file, delimiter=',')
    mat = mat[success, :]
    assert len(composed_phrases) == mat.shape[0]  # same number of rows

    # do the actual writing
    write_vectors_to_disk(
        sp.coo_matrix(mat),
        composed_phrases,
        ['RAE-feat%d' % i for i in range(100)],  # Socher provides 100-dimensional vectors
        vectors_file,
        gzipped=True)

    return vectors_file


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', choices=('fancy-compose', 'format', 'simple-compose'), nargs='+', required=True,
                        help="""Stages are as follows:
                         - fancy-compose: runs Socher's code (Turian unigrams and Socher composition)
                         - format: converts output of previous stage to Byblo-compatible files
                         - simple-compose: does Add/Mult... composition on Turian unigrams, as converted in
                         previous stage
                        """)
    args = parser.parse_args()

    if 'fancy-compose' in args.stages:
        run_socher_code()
    if 'format' in args.stages:
        reformat_socher_vectors()
    if 'simple-compose' in args.stages:
        composers = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer,
                     RightmostWordComposer, VerbComposer]
        compose_and_write_vectors(vectors_file,
                                  'turian',
                                  composers,
                                  output_dir=output_dir, gzipped=False, dense_hd5=True)