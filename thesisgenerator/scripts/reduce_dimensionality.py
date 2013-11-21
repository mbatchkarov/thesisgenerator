import logging
import sys
from sklearn.decomposition import TruncatedSVD

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from thesisgenerator.plugins.thesaurus_loader import Thesaurus
import numpy, scipy, time
from thesisgenerator.scripts import dump_all_composed_vectors as dump
from thesisgenerator.plugins.tokenizers import DocumentFeature
from thesisgenerator.composers.utils import write_vectors_to_disk
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s\t%(module)s.%(funcName)s ""(line %(lineno)d)\t%(levelname)s : %(""message)s")


def do_work(vector_file_paths,
            word_limits=[('N', 8), ('V', 4), ('J', 4), ('RB', 2)],
            reduce_to=[3, 10, 15]):
    thesaurus = Thesaurus(vector_file_paths)
    logging.info('Converting thesaurus to sparse matrix')
    mat, cols, rows = thesaurus.to_sparse_matrix()
    logging.info('Loaded a data matrix of shape %r', mat.shape)

    features = [DocumentFeature.from_string(r) for r in rows] # convert to document feature for access to PoS tag

    # don't want to do dimensionality reduction on composed vectors
    feature_types = [sorted_idx_and_pos_matching.type for sorted_idx_and_pos_matching in features]
    assert all(x == '1-GRAM' for x in feature_types)

    # get the PoS tags of each row in the matrix
    pos_tags = np.array([df.tokens[0].pos for df in features])
    print mat.sum()

    # find the rows of the matrix that correspond to the most frequent nouns, verbs, ...,
    # as measured by sum of feature counts. This is Byblo's definition of frequency (which is in fact a marginal),
    # but it is strongly correlated with one normally thinks of as entry frequency
    pos_to_rows = {}
    for desired_pos, desired_count in word_limits:
        row_of_current_pos = pos_tags == desired_pos # what rows are the right PoS tags at, boolean mask array
        # indices of the array sorted by row sum, and where the pos == desired_pos
        sorted_idx_by_sum = np.ravel(mat.sum(1)).argsort()
        row_of_current_pos = row_of_current_pos[sorted_idx_by_sum]
        sorted_idx_and_pos_matching = sorted_idx_by_sum[row_of_current_pos]
        # slice off the top desired_count and store them
        pos_to_rows[desired_pos] = sorted_idx_and_pos_matching[-desired_count:]
        logging.info('Frequency filter keeping %d/%d %s entries ', desired_count,
                     sum(row_of_current_pos), desired_pos)
    desired_rows = np.sort(np.hstack(x for x in pos_to_rows.values()))

    # check that the pos tag of each selected entry is what we think it is
    for k, v in pos_to_rows.iteritems():
        assert all(k == x for x in pos_tags[v])

    # remove the vectors for infrequent entries, update list of pos tags too
    mat = mat[desired_rows, :]
    rows = np.array(rows)[desired_rows]
    pos_tags = pos_tags[desired_rows]

    # removing rows may empty some columns, remove these as well. This is probably not very like to occur as we have
    # already filtered out infrequent features, so the column count will stay roughly the same
    desired_cols = np.ravel(mat.sum(0)) > 0
    mat = mat[:, desired_cols]
    cols = np.array(cols)[desired_cols]
    logging.info('Selected only the most frequent entries, matrix size is now %r', mat.shape)

    for n_components in reduce_to:
        if n_components > mat.shape[1]:
            logging.error('Cannot reduce dimensionality from %d to %d', mat.shape[1], n_components)
            continue

        method = TruncatedSVD(n_components)
        logging.info('Reducing dimensionality of matrix of shape %r', mat.shape)
        start = time.time()
        reduced_mat = method.fit_transform(mat)
        end = time.time()
        logging.info('Reduced using {} from shape {} to shape {} in {} seconds'.format(method,
                                                                                       mat.shape,
                                                                                       reduced_mat.shape,
                                                                                       end - start))

        for path, desired_pos in zip(dump.giga_paths, ['N', 'V', 'J', 'RB']):
            basename = path.split('.')[0]
            basename += '-SVD{}'.format(n_components)
            features_file = basename + '.features.filtered.strings'
            events_file = basename + '.events.filtered.strings'
            entries_file = basename + '.entries.filtered.strings'

            logging.info('Writing reduced vector files for PoS {}'.format(desired_pos))
            rows_for_this_pos = pos_tags == desired_pos
            tmp_mat = scipy.sparse.coo_matrix(reduced_mat[rows_for_this_pos, :])
            write_vectors_to_disk(tmp_mat, numpy.array(features)[rows_for_this_pos],
                                  ['SVD:feat{0:05d}'.format(i) for i in range(n_components)],
                                  features_file, entries_file, events_file)
            #'tmp/{}-svd{}.features.txt'.format(desired_pos, n_components),
            #'tmp/{}-svd{}.entries.txt'.format(desired_pos, n_components),
            #'tmp/{}-svd{}.events.txt'.format(desired_pos, n_components))


if __name__ == '__main__':
    do_work(dump.giga_paths,
            word_limits=[('N', 8000), ('V', 4000), ('J', 4000), ('RB', 200)],
            reduce_to=[300, 1000, 5000])
