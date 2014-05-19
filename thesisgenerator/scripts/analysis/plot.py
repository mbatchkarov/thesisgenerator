import logging
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from .utils import class_pull_results_as_list
import statsmodels.api as sm

#####################################################################
# FUNCTIONS THAT DISPLAY INFORMATION (PLOT OR LOG TO FILE)
#####################################################################

def print_counts_data(counts_objects, title):
    if not all(counts_objects):
        return  # did not collect any data
    logging.info('----------------------')
    logging.info('| %s time statistics:' % title)
    for field in counts_objects[0].__dict__:
        logging.info('| %s: mean %2.1f, std %2.1f', field,
                     np.mean([getattr(x, field) for x in counts_objects]),
                     np.std([getattr(x, field) for x in counts_objects]))
    logging.info('----------------------')


def histogram_from_list(l, subplot, title, weights=None):
    MAX_LABEL_COUNT = 40
    plt.subplot(2, 3, subplot)
    if type(l[0]) == str:
        # numpy's histogram doesn't like strings
        s = pd.Series(Counter(l))
        s.plot(kind='bar', rot=0, title=title)
    else:
        plt.hist(l, bins=MAX_LABEL_COUNT, weights=weights)
        plt.title(title)


def plot_dots(x, y, thickness, minsize=10., maxsize=200., draw_axes=True,
              xlabel='Class association of decode-time feature',
              ylabel='Class association of replacements'):
    z = np.array(thickness)
    range = min(z), max(z)
    if min(z) < minsize:
        z += (minsize - min(z))

    # http://stackoverflow.com/a/17029736/419338
    normalized_z = ((maxsize - minsize) * (z - min(z))) / (max(z) - min(z)) + minsize

    plt.scatter(x, y, normalized_z)
    if draw_axes:
        plt.hlines(0, min(x), max(x))
        plt.vlines(0, min(y), max(y))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return range


def plot_regression_line(x, y, weights=None):
    assert len(x) == len(y)
    if weights is None:
        weights = [1] * len(x)
    xs = np.linspace(min(x), max(x))
    x1 = sm.add_constant(x)
    model = sm.WLS(y, x1, weights=weights)
    results = model.fit()
    logging.info('Results of weighted linear regression: \n %s', results.summary())
    coef = results.params[::-1]  # statsmodels' linear equation is b+ax, numpy's is ax+b
    plt.plot(xs, results.predict(sm.add_constant(xs)), 'r-')
    return coef, results.rsquared, results.rsquared_adj


def sum_of_squares_score_diagonal_line(x, y, weights=None):
    assert len(x) == len(y)
    if weights is None:
        weights = [1] * len(x)
    else:
        assert len(weights) == len(x)

    #  check that the weights are all integers, doesn't make sense to pass them into np.repeat otherwise
    assert 0 == sum(weights - np.array(weights, dtype=int))

    x1 = np.repeat(x, weights)
    y1 = np.repeat(y, weights)

    return np.sum((x1 - y1) ** 2) / float(len(x1))

