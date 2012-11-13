'''
Created on Oct 19, 2012

@author: ml249
'''

def recall(confusion_mtx):
    try:
        return confusion_mtx['tp'] / (
        confusion_mtx['tp'] + confusion_mtx['fn'] + 0.0)
    except ZeroDivisionError:
        return 1


def specificity(confusion_mtx):
    try:
        return confusion_mtx['tn'] / (
        confusion_mtx['tn'] + confusion_mtx['fp'] + 0.0)
    except ZeroDivisionError:
        return 1