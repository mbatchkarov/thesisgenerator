import sys
from joblib import delayed
from joblib import Parallel

sys.path.append('.')
from discoutils.thesaurus_loader import Vectors
from collections import Counter
from discoutils.tokens import DocumentFeature
from thesisgenerator.utils import db
import pandas as pd


def info(**kwargs):
    stats = {}
    stats.update({'param_%s_' % k: v for k, v in kwargs.items()})
    try:
        v = Vectors.from_tsv(kwargs['path'])
        keys = [DocumentFeature.from_string(x) for x in v.keys()]

        type_counts = Counter(x.type for x in keys)
        stats.update({'%s_count' % k: v for k, v in type_counts.items()})

        pos_counts = Counter(x.tokens[0].pos for x in keys if x.type == '1-GRAM')
        stats.update({'%s_count' % k: v for k, v in pos_counts.items()})
    finally:
        return stats


if __name__ == '__main__':
    res = Parallel(n_jobs=2)(delayed(info)(**v._data) for v in db.Vectors.select())  # todo run for all vectors
    pd.DataFrame(res).to_csv('coverage_stats.csv')
