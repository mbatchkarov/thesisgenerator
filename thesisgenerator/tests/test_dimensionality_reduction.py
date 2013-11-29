import os

from operator import itemgetter
import pytest
import numpy as np
import scipy.sparse as sp

from thesisgenerator.plugins.thesaurus_loader import Thesaurus
from thesisgenerator.scripts.reduce_dimensionality import _do_svd_single, \
    _filter_out_infrequent_entries, _write_to_disk


DIM = 100
__author__ = 'mmb28'


@pytest.fixture(scope='module')
def dense_matrix():
    a = np.random.random((DIM, DIM))
    a[a < 0.4] = 0
    return a


@pytest.fixture(scope='module')
def sparse_matrix(dense_matrix):
    matrix = sp.csr_matrix(dense_matrix)
    assert matrix.nnz < DIM ** 2
    return matrix


def test_do_svd_single_dense(dense_matrix):
    for i in range(10, 51, 10):
        reducer, matrix = _do_svd_single(dense_matrix, i)
        matrix1 = reducer.inverse_transform(matrix)
        assert matrix.shape == (DIM, i)
        assert matrix1.shape == dense_matrix.shape


def test_do_svd_single_sparse(sparse_matrix):
    test_do_svd_single_dense(sparse_matrix)


@pytest.fixture(scope='module')
def all_cols(thesaurus_c):
    _, cols, _ = thesaurus_c.to_sparse_matrix()
    assert len(cols) == 5
    return cols


@pytest.mark.parametrize(
    ('feature_type_limits', 'expected_shape', 'missing_columns'),
    (
        ([('N', 2), ('V', 2), ('J', 2), ('AN', 2)], (5, 5), []), # nothing removed
        ([('N', 1), ('V', 2), ('J', 2), ('AN', 2)], (4, 5), []), # just row a/N should drop out
        ([('N', 0), ('V', 2), ('J', 2), ('AN', 2)], (3, 4), ['x/X']), # rows a and g, column x should drop out
        ([('V', 1)], (1, 3), ['b/V', 'x/X']), # just the one verb should remain, with its three features
    ),
)
def test_filter_out_infrequent_entries(thesaurus_c, all_cols, feature_type_limits, expected_shape, missing_columns):
    mat, pos_tags, rows, cols = _filter_out_infrequent_entries(feature_type_limits, thesaurus_c)
    assert mat.shape == expected_shape
    assert set(all_cols) - set(missing_columns) == set(cols)


def _read_and_strip_lines(input_file):
    with open(input_file) as infile:
        lines = infile.readlines()
    lines = map(str.strip, lines)
    lines = [x for x in lines if x]
    return lines


def test_write_to_file(tmpdir, thesaurus_c):
    '''
    Test writing thesauri containing one feature type in separate directories
    '''
    type_limits = sorted([('AN', 1), ('J', 1), ('N', 2), ('V', 1), ], key=itemgetter(0))
    matrix, pos_tags, rows, cols = _filter_out_infrequent_entries(
        type_limits,
        thesaurus_c)

    pos_per_output_dir = sorted(list(set(pos_tags)))
    output_prefixes = [str(tmpdir.join('%s.out' % x)) for x in pos_per_output_dir]
    _write_to_disk(matrix, None, output_prefixes, pos_per_output_dir, pos_tags, rows)

    for (type, max_count), prefix in zip(type_limits, output_prefixes):
        events_file = '%s.events.filtered.strings' % prefix
        assert os.path.exists(events_file)

        #check number of entries matches
        t1 = Thesaurus([events_file])
        assert len(t1) == max_count

        # check if the entries file has the right number of entries
        entries_file = '%s.entries.filtered.strings' % prefix
        assert os.path.exists(entries_file)
        assert len(_read_and_strip_lines(entries_file)) == len(t1)

        # check if the fetures file has the right number of features
        features_file = '%s.features.filtered.strings' % prefix
        assert os.path.exists(features_file)
        lines = _read_and_strip_lines(features_file)
        assert len(lines) <= matrix.shape[1] # some features might drop out because of 0 values, but in
        # any case there cannot be more features than dimensions in the matrix
