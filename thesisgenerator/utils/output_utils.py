"""
A set of utilities for manipulating/querying the output of thesisgenerator's classification experiments
"""
import itertools
from operator import attrgetter
import numpy as np
from thesisgenerator.utils import db

METRIC_DB = 'macrof1'
METRIC_CSV_FILE = 'macroavg_f1'


def get_single_vectors_field(exp_id, field_name):
    vectors = db.ClassificationExperiment.get(id=exp_id).vectors
    return getattr(vectors, field_name) if vectors else None


def get_cv_fold_count(ids):
    return [db.FullResults.select().where(db.FullResults.id == id).count() // 2 for id in ids]


def get_vectors_field(exp_ids, field_name):
    return np.repeat([get_single_vectors_field(exp_id, field_name) for exp_id in exp_ids],
                     get_cv_fold_count(exp_ids))
    return list(itertools.chain.from_iterable(x))


def get_cv_scores_single_experiment(n, classifier):
    rows = db.FullResults.select().where(db.FullResults.id == n,
                                      db.FullResults.classifier == classifier)
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

