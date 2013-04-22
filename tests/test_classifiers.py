from unittest import TestCase

from numpy import zeros

from classifiers import MostCommonLabelClassifier


__author__ = 'mmb28'


class TestMostCommonLabelClassifier(TestCase):
    def setUp(self):
        self.cl = MostCommonLabelClassifier()
        self.y = [1, 1, 1, 1, 0, 0]

    def test_predict(self):
        self.cl.fit(None, self.y)
        self.assertEqual(self.cl.decision, 1)

        y = self.cl.predict(zeros((3, 3)))
        self.assertListEqual(y.tolist(), [1, 1, 1])