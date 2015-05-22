import pytest
import numpy as np
import pandas as pd
from thesisgenerator.plugins.kmeans_disco import KmeansVectorizer, cluster_vectors


@pytest.fixture
def corpus():
    # because A and E belong to the same cluster, seeing A in a document
    # is equivalent to seen and E, and vice versa. The same goes for B and F.
    # Try a few combinations of these "words" in a document, they should all
    # be equivalent
    return [
        ['a/N', 'b/V'],
        ['e/N', 'b/V'],
        ['a/N', 'f/V'],
        ['e/N', 'f/V'],
        # ------------
        ['c/J', 'd/N'],
        ['g/J', 'd/N'],
        ['c/J', 'h/N'],
        ['g/J', 'h/N'],

    ]


@pytest.fixture
def clusters(tmpdir):
    vector_path = 'thesisgenerator/resources/twos.vectors.txt'
    put_path = str(tmpdir.join('clusters_unit_test.hdf'))
    cluster_vectors(vector_path, put_path, n_clusters=4, n_jobs=1)

    clusters = pd.read_hdf(put_path, key='clusters').clusters
    for i in range(4):
        # a and e, b and f, etc,  belong to the same cluster
        assert clusters[i] == clusters[i + 4]
    return put_path


def test_kmeans_vectorizer(corpus, clusters):
    feature_types = {'extract_unigram_features': set('JVN'), 'extract_phrase_features': []}
    v = KmeansVectorizer(clusters, min_df=0,
                         train_time_opts=feature_types,
                         decode_time_opts=feature_types)
    X, _ = v.fit_transform(corpus)

    assert X.shape == (8, 4)
    print(X.A)
    assert list(X.sum(axis=1).A1) == [2] * 8  # num feats in each document
    # first four document's vectors are all equal to each other, and so are the last four
    for i in range(3):
        np.testing.assert_array_equal(X.A[i, :], X.A[i + 1, :])
        np.testing.assert_array_equal(X.A[i + 4, :], X.A[i + 5, :])
