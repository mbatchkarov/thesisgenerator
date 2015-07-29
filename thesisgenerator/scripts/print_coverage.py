import os
import sys
import logging
from joblib import delayed
from joblib import Parallel

sys.path.append('.')
from discoutils.thesaurus_loader import Vectors
from collections import Counter, defaultdict
from discoutils.tokens import DocumentFeature
from thesisgenerator.utils import db
from thesisgenerator.scripts.extract_NPs_from_labelled_data import get_all_NPs_VPs
import pandas as pd


def info(**kwargs):
    stats = {}
    stats.update({'param_%s' % k: v for k, v in kwargs.items() if k not in ['path', 'size', 'modified', 'format']})

    if not os.path.exists(kwargs['path']):
        return stats

    v = Vectors.from_tsv(kwargs['path'])
    counts = get_all_NPs_VPs(return_counts=True, include_unigrams=True)
    feats_in_lab = set(counts.keys())

    logging.info('recording stats')
    keys = [DocumentFeature.from_string(x) for x in v.keys()]
    del v
    keys_in_labelled = [DocumentFeature.from_string(x) for x in keys if x in feats_in_lab]

    weighted_counts = defaultdict(int)
    for df in keys_in_labelled:
        if df.type == '1-GRAM':
            weighted_counts[df.type] += counts[df]
        else:
            weighted_counts[df.tokens[0].pos] += counts[df]
    stats.update({'%s_count_weighted' % k: v for k, v in weighted_counts.items()})

    type_counts = Counter(x.type for x in keys)
    stats.update({'%s_count_total' % k: v for k, v in type_counts.items()})
    # type_counts_lab = Counter(x.type for x in keys_in_labelled)
    # stats.update({'%s_count_in_labelled' % k: v for k, v in type_counts_lab.items()})

    pos_counts = Counter(x.tokens[0].pos for x in keys if x.type == '1-GRAM')
    pos_counts_lab = Counter(x.tokens[0].pos for x in keys_in_labelled if x.type == '1-GRAM')
    stats.update({'%s_count_total' % k: v for k, v in pos_counts.items()})
    stats.update({'%s_count_in_labelled' % k: v for k, v in pos_counts_lab.items()})
    return stats


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s """
                               "(line %(lineno)d)\t%(levelname)s : %(""message)s")
    vectors = db.Vectors.select().where((db.Vectors.rep == 0) & (db.Vectors.path != None))
    res = Parallel(n_jobs=2)(delayed(info)(**v._data) for v in vectors)
    df = pd.DataFrame(res)
    df.sort_index(axis=1).to_csv('coverage_stats.csv')
    logging.info('Done')
