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

    parser.add_argument('--data', choices=('labelled', 'unlabelled'), required=True,
                        help='Whether to zip labelled or unlabelled data sets. WARNING: Unlabelled requires '
                             'the database to have been populated with a list of vectors files)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true', default=False,
                       help='Whether to compress ALL available labelled/unlabelled '
                            'data sets or just one at a time')

    group.add_argument('--id', type=int,
                       help='If labelled data, compress just the labelled corpus at this position '
                            'in the predefined list. Otherwise compress just '
                            'this thesaurus id in the database')

    parameters = parser.parse_args()
    if parameters.data == 'labelled':
        if parameters.all:
            jsonify_all_labelled_corpora(parameters.jobs)
        else:
            jsonify_single_labelled_corpus(get_all_corpora()[parameters.id])
    else:
        if parameters.all:
            gzip_all_thesauri(parameters.jobs)
        else:
            gzip_single_thesaurus(Vectors.get(Vectors.id == parameters.id).path)