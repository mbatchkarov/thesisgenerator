# coding=utf-8
from unittest import TestCase
import os

import pytest
import numpy as np
from numpy.testing import assert_array_equal

from thesisgenerator.plugins import thesaurus_loader
from thesisgenerator.plugins.thesaurus_loader import _smart_lower, Thesaurus
from thesisgenerator.utils.misc import walk_nonoverlapping_pairs


__author__ = 'mmb28'


@pytest.fixture
def thesaurus_c():
    return Thesaurus(thesaurus_files=['thesisgenerator/resources/exp0-0c.strings'],
                     sim_threshold=0,
                     include_self=False,
                     ngram_separator='_')


def test_loading_bigram_thesaurus(thesaurus_c):
    assert len(thesaurus_c) == 5
    assert 'a/J_b/N' in thesaurus_c.keys()

# todo check this
def _assert_matrix_of_thesaurus_c_is_as_expected(matrix, rows, cols):
    # rows may come in any order
    assert set(rows) == set(['g/N', 'a/N', 'd/J', 'b/V', 'a/J_b/N'])
    # columns must be in alphabetical order
    assert cols == ['a/N', 'b/V', 'd/J', 'g/N', 'x/X']
    # test the vectors for each entry
    expected_matrix = np.array([
        [0.1, 0., 0.2, 0.8, 0.], # ab
        [0., 0.1, 0.5, 0.3, 0.], # a
        [0.1, 0., 0.3, 0.6, 0.], # b
        [0.5, 0.3, 0., 0.7, 0.], # d
        [0.3, 0.6, 0.7, 0., 0.9] # g
    ])
    # put the rows in the matrix in the order in which they are in expected_matrix
    matrix_ordered_by_rows = matrix[np.argsort(np.array(rows)), :]
    assert_array_equal(matrix_ordered_by_rows, expected_matrix)


def test_to_sparse_matrix(thesaurus_c):
    matrix, cols, rows = thesaurus_c.to_sparse_matrix()
    matrix = matrix.A
    assert matrix.shape == (5, 5)

    _assert_matrix_of_thesaurus_c_is_as_expected(matrix, rows, cols)


def test_to_dissect_core_space(thesaurus_c):
    """
    :type thesaurus_c: Thesaurus
    """
    space = thesaurus_c.to_dissect_core_space()
    matrix = space.cooccurrence_matrix.mat.A
    _assert_matrix_of_thesaurus_c_is_as_expected(matrix, space.id2row, space.id2column)


def test_to_file(thesaurus_c, tmpdir):
    """

    :type thesaurus_c: Thesaurus
    :type tmpdir: py.path.local
    """
    filename = str(tmpdir.join('outfile.txt'))
    thesaurus_c.to_file(filename)
    t1 = Thesaurus([filename])

    # can't just assert t1 == thesaurus_c, because to_file may reorder the columns
    for k, v in thesaurus_c.iteritems():
        assert k in t1.keys()
        assert set(v) == set(thesaurus_c[k])


def test_to_dissect_sparse_files(thesaurus_c, tmpdir):
    """

    :type thesaurus_c: Thesaurus
    :type tmpdir: py.path.local
    """
    from composes.semantic_space.space import Space

    prefix = str(tmpdir.join('output'))
    thesaurus_c.to_dissect_sparse_files(prefix)
    # check that files are there
    for suffix in ['sm', 'rows', 'cols']:
        outfile = '{}.{}'.format(prefix, suffix)
        assert os.path.exists(outfile)
        assert os.path.isfile(outfile)

    # check that reading the files in results in the same matrix
    space = Space.build(data="{}.sm".format(prefix),
                        rows="{}.rows".format(prefix),
                        cols="{}.cols".format(prefix),
                        format="sm")

    matrix, rows, cols = space.cooccurrence_matrix.mat, space.id2row, space.id2column
    exp_matrix, exp_cols, exp_rows = thesaurus_c.to_sparse_matrix()

    assert exp_cols == cols
    assert exp_rows == rows
    assert_array_equal(exp_matrix.A, matrix.A)
    _assert_matrix_of_thesaurus_c_is_as_expected(matrix.A, rows, cols)
    _assert_matrix_of_thesaurus_c_is_as_expected(exp_matrix.A, exp_rows, exp_cols)


class TestLoad_thesauri(TestCase):
    def setUp(self):
        """
        Sets the default parameters of the tokenizer and reads a sample file
        for processing
        """

        self.params = {
            'thesaurus_files': ['thesisgenerator/resources/exp0-0a.strings'],
            'sim_threshold': 0,
            # 'k': 10,
            'include_self': False
        }
        self.thesaurus = thesaurus_loader.Thesaurus(**self.params)

    def _reload_thesaurus(self):
        self.thesaurus = thesaurus_loader.Thesaurus(**self.params)

    def test_empty_thesaurus(self):
        self.params['thesaurus_files'] = []
        self._reload_thesaurus()
        self._reload_and_assert(0, 0)

        # should raise KeyError for unknown tokens
        with self.assertRaises(KeyError):
            self.thesaurus['kasdjhfka']

    def _reload_and_assert(self, entry_count, neighbour_count):
        th = thesaurus_loader.Thesaurus(**self.params)
        all_neigh = [x for v in th.values() for x in v]
        self.assertEqual(len(th), entry_count)
        self.assertEqual(len(all_neigh), neighbour_count)
        return th

    def test_sim_threshold(self):
        for i, j, k in zip([0, .39, .5, 1], [7, 3, 3, 0], [14, 4, 3, 0]):
            self.params['sim_threshold'] = i
            self._reload_thesaurus()
            self._reload_and_assert(j, k)


    def test_include_self(self):
        for i, j, k in zip([False, True], [7, 7], [14, 21]):
            self.params['include_self'] = i
            self._reload_thesaurus()
            th = self._reload_and_assert(j, k)

            for entry, neighbours in th.items():
                self.assertIsInstance(entry, str)
                self.assertIsInstance(neighbours, list)
                self.assertIsInstance(neighbours[0], tuple)
                if i:
                    self.assertEqual(entry, neighbours[0][0])
                    self.assertEqual(1, neighbours[0][1])
                else:
                    self.assertNotEqual(entry, neighbours[0][0])
                    self.assertGreaterEqual(1, neighbours[0][1])

    def test_smart_lower(self):
        # test that the PoS of an n-gram entry is not lowercased
        self.assertEquals(_smart_lower('Cat/N'), 'cat/N')
        self.assertEquals(_smart_lower('Cat/n'), 'cat/n')
        self.assertEquals(_smart_lower('Red/J_CaT/N'), 'red/J_cat/N')
        self.assertEquals(_smart_lower('Red/J CaT/N', separator=' '), 'red/J cat/N')

        # test that features are not touched
        self.assertEquals(_smart_lower('amod-DEP:former', aggressive_lowercasing=False), 'amod-DEP:former')

    def test_iterate_nonoverlapping_pairs(self):
        inp = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        output1 = [x for x in walk_nonoverlapping_pairs(inp, 1)]
        self.assertListEqual([(1, 2), (3, 4), (5, 6), (7, 8)], output1)