import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
import argparse
import logging
from thesisgenerator.utils.data_utils import (gzip_all_thesauri, jsonify_all_labelled_corpora,
                                              gzip_single_thesaurus, jsonify_single_labelled_corpus)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %("
                               "message)s")

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--jobs', type=int, help='Number of caching jobs (will do all possible thesauri and datasets)')
    group.add_argument('--experiment', type=int, help='Shelve just the thesaurus of this experiment')

    parameters = parser.parse_args()
    if parameters.jobs:
        gzip_all_thesauri(parser.parse_args().jobs)
        jsonify_all_labelled_corpora(parser.parse_args().jobs)
    else:
        gzip_single_thesaurus('conf/exp{0}/exp{0}_base.conf'.format(parameters.experiment))
        jsonify_single_labelled_corpus('conf/exp{0}/exp{0}_base.conf'.format(parameters.experiment))
