"""
A set of utilities for manipulating/querying the output of thesisgenerator's classification experiments
"""
import logging
import os
import itertools
from operator import attrgetter
import pandas as pd
from thesisgenerator.utils.db import ClassificationExperiment, FullResults

METRIC_DB = 'macrof1'
METRIC_CSV_FILE = 'macroavg_f1'


def get_single_vectors_field(exp_id, field_name):
    vectors = ClassificationExperiment.get(id=exp_id).vectors
    return getattr(vectors, field_name) if vectors else None


def get_vectors_field(exp_ids, field_name, cv_folds=25):
    x = [[get_single_vectors_field(exp_id, field_name)] * cv_folds for exp_id in exp_ids]
    return list(itertools.chain.from_iterable(x))


def get_cv_scores_single_experiment(n, classifier):
    rows = FullResults.select().where(FullResults.id == n,
                                      FullResults.classifier == classifier)
    rows = sorted(rows, key=attrgetter('cv_fold'))
    return [getattr(x, METRIC_CSV_FILE) for x in rows]

def get_scores(exp_ids, classifier='MultinomialNB'):
    data = []
    folds = []
    success = []
    for exp_number in exp_ids:
        scores = get_cv_scores_single_experiment(exp_number, classifier)
        if scores:
            cv_folds = len(scores)
            folds.extend(range(cv_folds))
            data.extend(scores)
            success.append(exp_number)
    return data, folds, success


# data, folds = get_scores([11, 12])
# reps = get_vectors_field([11, 12], 'rep')
# composers = get_vectors_field([11, 12], 'composer')
# percent = get_vectors_field([11, 12], 'unlabelled_percentage')

