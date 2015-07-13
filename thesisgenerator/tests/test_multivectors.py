from thesisgenerator.plugins.multivectors import MultiVectors
from thesisgenerator.tests.test_vector_sources import ones_vectors


def test_multivectors(ones_vectors):
    assert len(ones_vectors) == 4

    mv = MultiVectors([ones_vectors] * 3)
    assert 'a/N' in mv
    assert 'a/NP' not in mv

    for entry in ones_vectors.keys():
        n1 = [foo[0] for foo in ones_vectors.get_nearest_neighbours(entry)]
        n2 = [foo[0] for foo in mv.get_nearest_neighbours(entry)]
        assert n1 == n2

        sims = [foo[1] for foo in mv.get_nearest_neighbours(entry)]
        assert sims == [1, 1/2, 1/3, 1/4]
    assert mv.get_nearest_neighbours('asdf/N') is None
