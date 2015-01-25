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
from discoutils.misc import temp_chdir, mkdirs_if_not_exists, Bunch
from thesisgenerator.composers.vectorstore import (MultiplicativeComposer, AdditiveComposer,
                                                   RightmostWordComposer, LeftmostWordComposer)
from thesisgenerator.scripts.dump_all_composed_vectors import compose_and_write_vectors
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
                                  'NPs_in_R2_MR_tech_am_maas',
                                  'r2-mr-technion-am-maas-ANs-NNs-socher.txt')
socher_input_file = os.path.join(socher_base_dir, 'parsed.txt')

socher_output_phrases_file = os.path.join(socher_base_dir, 'phrases.txt')
socher_output_vectors_file = os.path.join(socher_base_dir, 'outVectors.txt')

# output of reformat stage
output_dir = os.path.join(socher_base_dir, 'composed')
vectors_file = os.path.join(output_dir, 'socher.events.filtered.strings')


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
    logging.info('Using phrases events file %s', socher_output_vectors_file)

    an_regex = re.compile("\(NP \(JJ (\S+)\) \(NN (\S+)\)\)\)")
    nn_regex = re.compile("\(NP \(NN (\S+)\) \(NN (\S+)\)\)\)")
    unigram_regex = re.compile("\(NP \((NN|JJ) (\S+)\)\)\)")

    # get a list of all phrases that we attempted to compose
    composed_phrases = []
    with open(socher_input_file) as infile:
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
    with open(socher_output_phrases_file) as infile:
        success = [i for i, line in enumerate(infile) if '*UNKNOWN*' not in line]
        # pick out just the phrases that composes successfully
    composed_phrases = itemgetter(*success)(composed_phrases)

    # load all vectors, remove these containing unknown words
    mat = np.loadtxt(socher_output_vectors_file, delimiter=',')
    mat = mat[success, :]
    assert len(composed_phrases) == mat.shape[0]  # same number of rows

    # CREATE BYBLO EVENTS/FEATURES/ENTRIES FILE FROM INPUT
    mkdirs_if_not_exists(output_dir)

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
                     RightmostWordComposer]
        # todo writing file there is messy but needs to be this way for backwards compat
        ngram_vectors_dir = os.path.join(prefix,
                                         'exp12-14-composed-ngrams')
        compose_and_write_vectors(vectors_file,
                                  'turian',  # todo this param is useless
                                  composers,
                                  output_dir=ngram_vectors_dir)