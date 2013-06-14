# coding=utf-8

"""
Added changes from https://github.com/scikit-learn/scikit-learn/pull/2036
so that they can be used before they are accepted into scikit-learn
"""
from itertools import count, izip
from sklearn.base import is_classifier, clone
from sklearn.cross_validation import check_cv
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils import check_arrays, safe_mask
import numpy as np
import scipy.sparse as sp

__author__ = 'mmb28'


def _cross_val_score(cv_number, estimator, X, y, score_func, train, test,
                     verbose, fit_params):
    """Inner loop for cross validation"""

    # set the cv_number on the estimator
    estimator.cv_number = cv_number
    # if the estimator is a pipeline, set cv_number on all of its components

    if hasattr(estimator, 'named_steps'):
        for _, est in estimator.named_steps.iteritems():
            est.cv_number = cv_number

    n_samples = X.shape[0] if sp.issparse(X) else len(X)
    fit_params = dict([(k, np.asarray(v)[train]
    if hasattr(v, '__len__')
    and len(v) == n_samples else v)
                       for k, v in fit_params.items()])
    if not hasattr(X, "shape"):
        if getattr(estimator, "_pairwise", False):
            raise ValueError("Precomputed kernels or affinity matrices have "
                             "to be passed as arrays or sparse matrices.")
        X_train = [X[idx] for idx in train]
        X_test = [X[idx] for idx in test]
    else:
        if getattr(estimator, "_pairwise", False):
            # X is a precomputed square kernel matrix
            if X.shape[0] != X.shape[1]:
                raise ValueError("X should be a square kernel matrix")
            X_train = X[np.ix_(train, train)]
            X_test = X[np.ix_(test, train)]
        else:
            X_train = X[safe_mask(X, train)]
            X_test = X[safe_mask(X, test)]

    if y is None:
        estimator.fit(X_train, **fit_params)
        if score_func is None:
            score = estimator.score(X_test)
        else:
            score = score_func(X_test)
    else:
        estimator.fit(X_train, y[train], **fit_params)
        if score_func is None:
            score = estimator.score(X_test, y[test])
        else:
            score = score_func(y[test], estimator.predict(X_test))
    if verbose > 1:
        print("score: %f" % score)
    return cv_number, score


def naming_cross_val_score(estimator, X, y=None, score_func=None, cv=None,
                           n_jobs=1,
                           verbose=0, fit_params=None):
    """Evaluate a score by cross-validation

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape at least 2D
        The data to fit.

    y : array-like, optional
        The target variable to try to predict in the case of
        supervised learning.

    score_func : callable, optional
        Score function to use for evaluation.
        Has priority over the score function in the estimator.
        In a non-supervised setting, where y is None, it takes the test
        data (X_test) as its only argument. In a supervised setting it takes
        the test target (y_true) and the test prediction (y_pred) as arguments.

    cv : cross-validation generator, optional
        A cross-validation generator. If None, a 3-fold cross
        validation is used or 3-fold stratified cross-validation
        when y is supplied and estimator is a classifier.

    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    verbose : integer, optional
        The verbosity level.

    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.

    Returns
    -------
    scores : array of float, shape=(len(list(cv)),)
        Array of scores of the estimator for each run of the cross validation.
    """
    X, y = check_arrays(X, y, sparse_format='csr', allow_lists=True)
    cv = check_cv(cv, X, y, classifier=is_classifier(estimator))
    if score_func is None:
        if not hasattr(estimator, 'score'):
            raise TypeError(
                "If no score_func is specified, the estimator passed "
                "should have a 'score' method. The estimator %s "
                "does not." % estimator)
            # We clone the estimator to make sure that all the folds are
            # independent, and that it is pickle-able.
    fit_params = fit_params if fit_params is not None else {}
    scores = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_cross_val_score)(cv_number,
                                  clone(estimator), X, y, score_func,
                                  train, test, verbose, fit_params)
        for (cv_number, (train, test)) in izip(count(), cv))
    return scores
