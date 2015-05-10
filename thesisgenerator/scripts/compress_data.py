import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
import argparse
import logging
from thesisgenerator.utils.db import Vectors
from thesisgenerator.utils.data_utils import (gzip_all_thesauri, jsonify_all_labelled_corpora,
                                              gzip_single_thesaurus, jsonify_single_labelled_corpus,
                                              get_all_corpora)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %("
                               "message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=int, default=4,
                        help='Number of concurrent jobs')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true', default=False,
                       help='Whether to compress ALL available labelled/unlabelled '
                            'data sets or just one at a time')

    group.add_argument('--id', type=int,
                       help='If labelled data, compress just the labelled corpus at this position '
                            'in the predefined list. If unlabelled compress just '
                            'this thesaurus id in the database (must have been populated)')

    parameters = parser.parse_args()
    if parameters.all:
        jsonify_all_labelled_corpora(parameters.jobs)
    else:
        jsonify_single_labelled_corpus(get_all_corpora()[parameters.id])