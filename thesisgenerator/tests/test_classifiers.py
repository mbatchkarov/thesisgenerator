# coding=utf-8
from collections import Counter
from numpy import zeros
from thesisgenerator.classifiers import MostCommonLabelClassifier, SubsamplingPredefinedIndicesIterator
import numpy as np


def test_predict():
    clf = MostCommonLabelClassifier()
    clf = clf.fit(None, [1, 1, 1, 1, 0, 0])
    assert clf.decision == 1

    y = clf.predict(zeros((3, 3)))
    assert y.tolist() == [1, 1, 1]


def test_SubsamplingPredefinedIndicesIterator():
    y_vals = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # chunk for training
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # chunk for testing
    train_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    test_indices = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]

    for num_samples in [1, 3, 5, 10, 20]:
        for sample_size in [2, 4, 6, 10, 20]:
            it = SubsamplingPredefinedIndicesIterator(y_vals, train_indices, test_indices, num_samples, sample_size)

            assert len(it) == num_samples
            for train, test in it:
                assert test == test_indices
                assert len(train) == sample_size
                assert all(x in train_indices for x in train)
                counts = Counter(y_vals[train])
                # equal number of positives and negatives in sample
                assert counts[0] == counts[1]

    it = SubsamplingPredefinedIndicesIterator(y_vals, train_indices, test_indices, 2, 5)
    for train, test in it:
        assert test == test_indices
        # one extra point needed to maintain 1:1 ratio of positives and negatives
        assert len(train) == 6
        assert all(x in train_indices for x in train)