# adapted from https://gist.github.com/adrn/3993992
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
from scipy.stats import gaussian_kde

def find_confidence_interval(x, pdf, confidence_level):
    return pdf[pdf > x].sum() - confidence_level

def density_contour(xdata, ydata, w, nbins_x, nbins_y, ax=None, **contour_kwargs):
    """ Create a density contour plot.

    Parameters
    ----------
    xdata : numpy.ndarray
    ydata : numpy.ndarray
    nbins_x : int
        Number of bins along x dimension
    nbins_y : int
        Number of bins along y dimension
    ax : matplotlib.Axes (optional)
        If supplied, plot the contour to this axis. Otherwise, open a new figure
    contour_kwargs : dict
        kwargs to be passed to pyplot.contour()
    """
    print 'histogram'
    H, xedges, yedges = np.histogram2d(xdata, ydata, bins=(nbins_x,nbins_y), weights=w)
    x_bin_sizes = (xedges[1:] - xedges[:-1]).reshape((1,nbins_x))
    y_bin_sizes = (yedges[1:] - yedges[:-1]).reshape((nbins_y,1))

    pdf = (H*(x_bin_sizes*y_bin_sizes))

    # one_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.68))
    # two_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.95))
    # three_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.99))
    # levels = [one_sigma, two_sigma, three_sigma]

    X, Y = 0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1])
    Z = pdf.T

    # data = np.hstack([X, Y, Z.diagonal()]).T
    # kernel = gaussian_kde(data)

    print 'plot'
    contour = plt.contour(X, Y, Z, origin="lower", **contour_kwargs)

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # contour = ax.plot_trisurf(X,Y,Z)


    # plt.scatter(xdata, ydata)

    return contour

def test_density_contour():
    print 'generating'
    norm = np.random.normal(10., 15., size=(12340, 2))
    density_contour(norm[:,0], norm[:,1], np.arange(12340), 100, 100)
    plt.show()

test_density_contour()