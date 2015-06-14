import logging
import pytest
import numpy as np
import pandas as pd
from thesisgenerator.plugins.kmeans_disco import KmeansVectorizer, cluster_vectors

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %(message)s")


@pytest.fixture(scope='module')
def corpus():
    # because A and E belong to the same cluster, seeing A in a document
    # is equivalent to seen and E, and vice versa. The same goes for B and F.
    # Try a few combinations of these "words" in a document, they should all
    # be equivalent
    return [
        ['a/N', 'b/V', 'not_in_vocabulary'],
        ['e/N', 'b/V'],
        ['a/N', 'f/V'],
        ['e/N', 'f/V'],
        # ------------
        ['c/J', 'd/N'],
        ['g/J', 'd/N'],
        ['c/J', 'h/N'],
        ['g/J', 'h/N'],
    ]


@pytest.fixture(scope='module')
def corpus_small():
    # some clusters (third and fourth) are not present in the corpus
    return [['a/N', 'b/V']]


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


def _vectorize(clusters, corpus):
    feature_types = {'extract_unigram_features': set('JVN'), 'extract_phrase_features': []}
    v = KmeansVectorizer(min_df=0,
                         train_time_opts=feature_types,
                         decode_time_opts=feature_types)
    X, _ = v.fit_transform(corpus, clusters=pd.read_hdf(clusters, key='clusters'))
    return X, v


def test_kmeans_vectorizer(corpus, corpus_small, clusters):
    X, vect = _vectorize(clusters, corpus)
    assert X.shape == (8, 4)
    print(X.A)
    assert list(X.sum(axis=1).A1) == [2] * 8  # num feats in each document
    # first four document's vectors are all equal to each other, and so are the last four
    for i in range(3):
        np.testing.assert_array_equal(X.A[i, :], X.A[i + 1, :])
        np.testing.assert_array_equal(X.A[i + 4, :], X.A[i + 5, :])

    X, _ = vect.transform(corpus_small)
    print(X.A)
    assert X.shape == (1, 4)

def test_kmeans_vectorizer_missing_clusters(corpus_small, clusters):
    # when a cluster is missing from the labelled corpus, it should not be added to the vocabulary
    # this will cause problems later
    X, _ = _vectorize(clusters, corpus_small)
    print(X.A)
    assert X.shape == (1, 2)

