import numpy as np
from numpy.testing import assert_almost_equal
import pytest
from thesisgenerator.scripts.analysis.plot import plot_regression_line, sum_of_squares_score_diagonal_line
from thesisgenerator.scripts.analysis.signified_internals_analysis import ReplacementLogOddsScore


def _eval_at(coef, x):
    """
    Evaluates ax + b, where a,b = coef
    :param coef: array-like of len 2, higher-order coefficient first
    :type x: scalar or array-like
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
    x1 = np.array([0, 1, 2, 3, 3])
    coef1, _, _ = plot_regression_line(x1, y1, w1)

    w2 = np.array([1, 1, 1, 2])
    y2 = np.array([2, 3, 4, 6])
    x2 = np.array(range(len(y2)))
    coef2, _, _ = plot_regression_line(x2, y2, w2)
    assert set(x1) == set(x2)
    assert_almost_equal(coef1, coef2)


def test_sum_of_squares_score_diagonal_line():
    # unequal parameter lengths
    with pytest.raises(AssertionError):
        sum_of_squares_score_diagonal_line([1], [1, 2])
    # non-integer weights
    with pytest.raises(AssertionError):
        sum_of_squares_score_diagonal_line([1, 2], [1, 2], [1.0, 1.1])

    # all points weighted equally
    # perfect fit
    assert sum_of_squares_score_diagonal_line([-1, 1, 2, 3], [-1, 1, 2, 3]) == 0

    # invariant to number of data points
    v1 = sum_of_squares_score_diagonal_line([-1, 1, 3], [-1, 1, 3])
    v2 = sum_of_squares_score_diagonal_line([-1, 1, 2, 3], [-1, 1, 2, 3])
    assert v1 == v2

    # a line that is further off 45 degrees has a higher error
    v1 = sum_of_squares_score_diagonal_line([-1, 1, 2, 3], [-1, 1, 2, 3])
    v2 = sum_of_squares_score_diagonal_line([-1, 1, 2, 3], [-1, 1, 2, 4])
    v3 = sum_of_squares_score_diagonal_line([-1, 1, 2, 3], [-1, 1, 2, 5])
    assert v1 < v2 < v3

    # check the weighted case
    # perfect fit, different weights
    v1 = sum_of_squares_score_diagonal_line([-1, 1, 3], [-1, 1, 3], [2, 1, 1])
    v2 = sum_of_squares_score_diagonal_line([-1, 1, 3], [-1, 1, 3], [1, 2, 2])
    assert v1 == v2

    # OK fit, increasing weight of point that is off target
    v0 = sum_of_squares_score_diagonal_line([-1, 3], [-1, 3], [1, 1])
    v1 = sum_of_squares_score_diagonal_line([-1, 3], [-1, 4], [1, 1])
    v2 = sum_of_squares_score_diagonal_line([-1, 3], [-1, 4], [1, 2])
    v3 = sum_of_squares_score_diagonal_line([-1, 3], [-1, 4], [1, 3])
    assert v0 < v1 < v2 < v3

def test_ReplacementLogOddsScore_add_up():
    # check counts are correctly added up
    assert ReplacementLogOddsScore.add_up([ReplacementLogOddsScore(1.1, 2.2, 1, 1)] * 5) == \
           [ReplacementLogOddsScore(1.1, 2.2, 5, 1)]

    # check similarities are averaged
    assert ReplacementLogOddsScore.add_up([
        ReplacementLogOddsScore(1.1, 2.2, 1, 1),
        ReplacementLogOddsScore(1.1, 2.2, 1, 3),
        ReplacementLogOddsScore(10, 22, 1, 1),
        ReplacementLogOddsScore(10, 22, 1, 3),
        ]) == \
           [
               ReplacementLogOddsScore(1.1, 2.2, 2, 2),
               ReplacementLogOddsScore(10, 22, 2, 2),
            ]