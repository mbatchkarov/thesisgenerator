import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from .utils import class_pull_results_as_list

#####################################################################
# FUNCTIONS THAT DISPLAY INFORMATION (PLOT OR LOG TO FILE)
#####################################################################

def print_counts_data(counts_objects, title):
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


def plot_dots(replacement_scores, minsize=10., maxsize=200., draw_axes=True,
              xlabel='Class association of decode-time feature',
              ylabel='Class association of replacements'):
    x, y, thickness = class_pull_results_as_list(replacement_scores)
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


def plot_regression_line(x, y, z):
    coef = np.polyfit(x, y, 1, w=z)
    xi = np.linspace(min(x), max(x))
    line = coef[0] * xi + coef[1]
    plt.plot(xi, line, 'r-')
    return coef