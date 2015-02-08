import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
import argparse
import logging
from os.path import join
import pandas as pd
from scipy import sparse as sp
from discoutils.cmd_utils import run_and_log_output
from discoutils.misc import temp_chdir
from discoutils.io_utils import write_vectors_to_disk
from thesisgenerator.composers.vectorstore import (RightmostWordComposer, LeftmostWordComposer,
                                                   MultiplicativeComposer, AdditiveComposer,
                                                   compose_and_write_vectors)


prefix = '/mnt/lustre/scratch/inf/mmb28/FeatureExtractionToolkit'
glove_dir = join(prefix, 'glove')
glove_script = join(glove_dir, 'demo_miro.sh')  # set param in that script
pos_only_data_dir = join(prefix, 'data/gigaword-afe-split-pos/gigaword/')
unlabelled_data = join(prefix, 'data/gigaword-afe-split-pos/gigaword.oneline')
raw_vectors_file = join(glove_dir, 'vectors.txt')  # what GloVe produces
formatted_vectors_file = join(glove_dir, 'vectors.miro.gz')  # unigram vectors in my format
composer_algos = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer, RightmostWordComposer]


def compute_and_write_vectors(stages):
    if 'reformat' in stages:
        # glove requires the entire corpus to be on a single row
        logging.info('Starting corpus reformat')
        run_and_log_output('cat {}/* > tmp'.format(pos_only_data_dir))
        run_and_log_output('tr "\\n" " " < tmp > {}'.format(unlabelled_data))
        run_and_log_output('rm -f tmp')
        logging.info('Done with reformat')

    if 'vectors' in stages:
        logging.info('Starting training')
        with temp_chdir(glove_dir):
            run_and_log_output('sh {} {}'.format(glove_script, unlabelled_data))
        logging.info('Done training, converting to Byblo-compatible gzip')

        # convert their format to ours
        mat = pd.read_csv(raw_vectors_file, sep=' ', index_col=0, header=0)
        cols = ['f%d' % i for i in range(mat.shape[1])]
        write_vectors_to_disk(sp.coo_matrix(mat.values), mat.index, cols, formatted_vectors_file, gzipped=True)

    if 'compose' in stages:
        logging.info('Loading labelled corpora and composing phrase vectors therein')
        compose_and_write_vectors(formatted_vectors_file,
                                  'glove-gigaw',
                                  composer_algos,
                                  output_dir=glove_dir)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', choices=('reformat', 'vectors', 'compose'),
                        required=True, nargs='+')
    args = parser.parse_args()
    compute_and_write_vectors(args.stages)

