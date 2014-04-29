# coding=utf-8
"""
Created on Oct 17, 2012

@author: ml249
"""

from sklearn.metrics import f1_score, precision_score, recall_score


def macroavg_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro', pos_label=None)


def macroavg_prec(y_true, y_pred):
    return precision_score(y_true, y_pred, average='macro', pos_label=None)


def macroavg_rec(y_true, y_pred):
    return recall_score(y_true, y_pred, average='macro', pos_label=None)


def microavg_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro', pos_label=None)


def microavg_prec(y_true, y_pred):
    return precision_score(y_true, y_pred, average='micro', pos_label=None)


def microavg_rec(y_true, y_pred):
    return recall_score(y_true, y_pred, average='micro', pos_label=None)
