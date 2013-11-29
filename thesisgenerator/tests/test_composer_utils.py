import pytest
import scipy.sparse as sp

from thesisgenerator.plugins.tokenizers import DocumentFeature
from thesisgenerator.tests.test_dimensionality_reduction import _read_and_strip_lines
from thesisgenerator.composers.utils import write_vectors_to_disk
from thesisgenerator.tests.test_thesaurus import thesaurus_c


__author__ = 'mmb28'


@pytest.fixture(params=['normal', 'abbreviated', 'empty', 'with_filter', 'with_verb_only_filter'], scope='function')
def resources(thesaurus_c, request):
    filter_callable = lambda x: x
    features = ['a/N', 'b/V', 'd/J', 'g/N', 'x/X']
    entries = list(sorted(thesaurus_c.keys()))

    if request.param == 'empty':
        thesaurus_c.clear()
        features, entries = [], []

    if request.param == 'with_filter':
        filter_callable = lambda x: False
        features, entries = [], []

    if request.param == 'with_verb_only_filter':
        filter_callable = lambda x: x.tokens[0].pos == 'V'
        features = ['a/N', 'd/J', 'g/N']
        entries = ['b/V']

    if request.param == 'abbreviated':
        del thesaurus_c['g/N']
        features.pop(-1) # remove x/X
        entries = thesaurus_c.keys()

    return thesaurus_c, entries, features, filter_callable


def test_write_vectors_to_disk(resources, tmpdir):
    """
    Checks the entries/features files, the events file is checked by
    thesisgenerator.tests.test_thesaurus.test_to_file

    :type th: Thesaurus
    """
    th, expected_entries, expected_features, filter_callable = resources
    events_file = str(tmpdir.join('events.txt'))
    entries_file = str(tmpdir.join('entries.txt'))
    features_file = str(tmpdir.join('features.txt'))

    matrix, cols, rows = th.to_sparse_matrix()
    rows = [DocumentFeature.from_string(x) for x in rows]
    write_vectors_to_disk(sp.coo_matrix(matrix), rows, cols,
                          features_file, entries_file, events_file,
                          entry_filter=filter_callable)

    entries = [x.split('\t')[0] for x in _read_and_strip_lines(entries_file)]
    features = [x.split('\t')[0] for x in _read_and_strip_lines(features_file)]

    assert set(entries) == set(expected_entries)
    assert features == expected_features