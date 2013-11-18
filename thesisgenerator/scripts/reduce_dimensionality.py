import logging
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from thesisgenerator.plugins.thesaurus_loader import Thesaurus
from sklearn.decomposition.nmf import NMF
import numpy, scipy, time
from thesisgenerator.scripts import dump_all_composed_vectors as dump
from thesisgenerator.plugins.tokenizers import DocumentFeature
from thesisgenerator.composers.utils import write_vectors_to_disk

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s\t%(module)s.%(funcName)s ""(line %(lineno)d)\t%(levelname)s : %(""message)s")


def do_work(paths):
    thesaurus = Thesaurus(paths)

    n_components = 10
    mat, cols, rows = thesaurus.to_sparse_matrix()
    features = [DocumentFeature.from_string(x) for x in rows]
    feature_types = [x.type for x in features]
    # don't want to do dimensionality reduction on composed vectors
    assert all(x == '1-GRAM' for x in feature_types)
    feature_pos = [x.tokens[0].pos for x in features]
    print mat.sum()

    for n_components in [300, 1000, 5000]:
        method = NMF(n_components)
        logging.info(mat.shape)
        start = time.time()
        reduced_mat = method.fit_transform(mat)
        end = time.time()
        logging.info('Reduced using {} from shape {} to shape {} in {} seconds'.format(method,
                                                                                       mat.shape,
                                                                                       reduced_mat.shape,
                                                                                       end - start))

        print reduced_mat.sum()
        for path, desired_pos in zip(dump.giga_paths, ['N', 'V', 'J', 'RB']):
            basename = path.split('.')[0]
            basename += '-nmf{}'.format(n_components)
            features_file = basename + '.features.filtered.strings'
            events_file = basename + '.events.filtered.strings'
            entries_file = basename + '.entries.filtered.strings'
            out_path = '.'.join(basename)

            logging.info('Writing reduces vector files for PoS {} to {}'.format(desired_pos, out_path))
            row_nums_to_include = [i for i, pos in enumerate(feature_pos) if pos == desired_pos]
            tmp_mat = scipy.sparse.coo_matrix(reduced_mat[row_nums_to_include, :])
            write_vectors_to_disk(tmp_mat, numpy.array(features)[row_nums_to_include],
                                  ['NMF:feat{0:05d}'.format(i) for i in range(n_components)],
                                  features_file, entries_file, events_file)
            #'tmp/{}-svd{}.features.txt'.format(desired_pos, n_components),
            #'tmp/{}-svd{}.entries.txt'.format(desired_pos, n_components),
            #'tmp/{}-svd{}.events.txt'.format(desired_pos, n_components))


if __name__ == '__main__':
    do_work(dump.giga_paths)