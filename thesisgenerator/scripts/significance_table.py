from pandas import DataFrame
from scipy.stats import ttest_rel
from pandas.io.parsers import read_csv
from pandas.io.sql import read_frame
import numpy as np
from thesisgenerator.utils.misc import get_susx_mysql_conn
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison


def format(y):
    try:
        x = float(y)
    except ValueError:
        # this must be a lable, leave it
        return y
    if np.isnan(x):
        return '--'
    if x < 0.01:
        return '**'
    elif x < 0.05:
        return ' *'
    else:
        return '  '  # not significant


def get_significance_table(experiments, classifier='MultinomialNB'):
    # get human-readable labels for the table
    gf = read_frame('SELECT * FROM ExperimentDescriptions', get_susx_mysql_conn())
    data = []
    composers = []
    # composers = [gf[gf['number'] == n]['composer'].tolist()[0] for n in experiments]

    for n in experiments:
        composer_name = gf[gf['number'] == n]['composer'].tolist()[0]  # human-readable name
        composers.extend([composer_name] * 25)

        outfile = 'conf/exp{0}/output/exp{0}-10.out-raw.csv'.format(n)  # only at size 500
        df = read_csv(outfile)
        mask = df['classifier'].isin([classifier]) & df['metric'].isin(['macroavg_f1'])
        ordered_scores = df['score'][mask].tolist()
        data.append(ordered_scores)

        # #plot distribution of scores to visually check for normality
        # plt.figure()
        # plt.hist(sorted(ordered_scores), bins=12)
        # plt.savefig('distrib%d.png' % n, format='png')
    data = np.hstack(data)
    mod = MultiComparison(data, composers, group_order=composers)
    a = mod.tukeyhsd(alpha=0.01)  # reject hypothesis that mean is the same? rej=true means a sign. difference exists
    print 'Group names are', dict(enumerate(composers[::25]))
    print 'Experiments are', experiments
    return a


if __name__ == '__main__':
    # define what experiments we're interested in
    # NB! make sure there are no repeated composers

    print get_significance_table([1, 2, 3, 4, 5, 57, 61])  # SVD0, R2, deps
    # print get_significance_table([23, 24, 25, 26, 27, 61])  # SVD0, R2, wins

    print get_significance_table([6, 7, 8, 9, 10, 11, 58, 61])  # SVD100, R2, deps
    print get_significance_table([28, 29, 30, 31, 32, 33, 61])  # SVD100, R2, wins

    # print get_significance_table([12, 13, 14, 15, 16, 59, 62])  # SVD0, MR, deps
    # print get_significance_table([34, 35, 36, 37, 38, 62])  # SVD0, MR, wins

    print get_significance_table([17, 18, 19, 20, 21, 22, 60, 62])  # SVD100,MR, deps
    # print get_significance_table([39, 40, 41, 42, 43, 44, 62])  # SVD100,MR, wins