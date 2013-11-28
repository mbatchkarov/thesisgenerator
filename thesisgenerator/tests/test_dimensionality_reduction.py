import pytest
import numpy as np
import scipy.sparse as sp
from thesisgenerator.scripts.reduce_dimensionality import _do_svd_single, _filter_out_infrequent_entries

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
    print 1
    for i in range(10, 51, 10):
        reducer, matrix = _do_svd_single(dense_matrix, i)
        matrix1 = reducer.inverse_transform(matrix)
        assert matrix.shape == (DIM, i)
        assert matrix1.shape == dense_matrix.shape


def test_do_svd_single_sparse(sparse_matrix):
    test_do_svd_single_dense(sparse_matrix)

# todo finish this- must work for all combinations of desired counts, including 0
def test_filter_out_infrequent_entries(thesaurus_c):
    feature_type_limits = [('N', 200), ('V', 200), ('J', 4000), ('AN', 200)]
    mat, pos_tags, rows = _filter_out_infrequent_entries(feature_type_limits, thesaurus_c)
    assert mat.shape == (5, 5) #nothing lost because limits are high

    feature_type_limits = [('N', 0), ('V', 4000), ('J', 4000), ('AN', 200)]
    mat, pos_tags, rows = _filter_out_infrequent_entries(feature_type_limits, thesaurus_c)
    assert len(rows) == len(thesaurus_c) # lost all nouns
    print mat.A, pos_tags, rows
    assert 0