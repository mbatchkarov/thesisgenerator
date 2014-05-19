import numpy as np
from numpy.testing import assert_almost_equal
from thesisgenerator.scripts.analysis.plot import plot_regression_line


def _eval_at(coef, x):
    """
    Evaluates ax + b, where a,b = coef
    :param coef: array-like of len 2, higher-order coefficient first
    :type coef:
    :param x:
    :type x: scalar or array-like
    :return:
    :rtype:
    """
    assert len(coef) == 2  # model must be linear
    return np.polyval(coef, x)


def test_weighted_regression():
    coef, _, _ = plot_regression_line(range(5), range(5))
    assert_almost_equal(coef, (1, 0))

    # the high weight on the x=3 drag the regression line down even down the first and last points pull it up
    y = np.array([1, 1, 2, 2, 4, 6])
    w = np.array([1, 1, 1, 14, 1, 1])
    x = np.array(range(6))
    coef, _, _ = plot_regression_line(x, y, w)
    assert _eval_at(coef, 3) < 3
    assert _eval_at(coef, 0) < 0
    assert _eval_at(coef, 5) < 5

    # when that weight is reduced the regression line goes up
    w = np.array([1, 1, 1, 1, 1, 1])
    coef, _, _ = plot_regression_line(x, y, w)
    assert _eval_at(coef, 3) > 3
    assert _eval_at(coef, 0) > 0
    assert _eval_at(coef, 5) > 5

    # test that increasing the weight of a data point gets the regression line closer to that point
    y = np.array([2, 3, 4, 6])
    x = np.array(range(len(y)))
    w1 = np.array([1, 1, 1, 1])
    coef1, _, _ = plot_regression_line(x, y, w1)
    w2 = np.array([1, 1, 1, 2])
    coef2, _, _ = plot_regression_line(x, y, w2)
    assert 6 - _eval_at(coef1, 3) > 6 - _eval_at(coef2, 3)


    # test that duplicating a data point is equivalent to doubling its weight
    # test that duplicating a data point is equivalent to doubling its weight
    w1 = np.array([1, 1, 1, 1, 1])
    y1 = np.array([2, 3, 4, 6, 6])
    x1 = np.array([0,1,2,3,3])
    coef1, _, _ = plot_regression_line(x1, y1, w1)

    w2 = np.array([1, 1, 1, 2])
    y2 = np.array([2, 3, 4, 6])
    x2 = np.array(range(len(y2)))
    coef2, _, _ = plot_regression_line(x2, y2, w2)
    assert set(x1) == set(x2)
    assert_almost_equal(coef1, coef2)

def test_sum_of_squares_score_diagonal_line():
    assert False