# coding=utf-8
from math import sqrt
import os
import errno

from jinja2 import Template
import matplotlib.pyplot as plt
import pandas.io.sql as psql

from thesisgenerator.utils.misc import get_susx_mysql_conn


coverage_sql = \
    "SELECT DISTINCT name, sample_size,total_tok,total_typ, " \
    "{{ variables|join(', ') }} FROM data{{ '{:02d}'.format(number)}} " \
    "{% if wheres %} WHERE {{ wheres|join(' and ') }} {% endif %} " \
    "ORDER BY sample_size"

perf_sql = "SELECT sample_size,score_mean,score_std " \
           "FROM data{{ '{:02d}'.format(number)}} " \
           "WHERE metric=\"macroavg_f1\" and classifier=\"{{classifier}}\"" \
           "{% if wheres %} and {{ wheres|join(' and ') }} {% endif %} " \
           "ORDER BY sample_size"


def query_to_data_frame(sql):
    return psql.frame_query(sql, get_susx_mysql_conn())


# <codecell>

def _safe_column_names(data_frames, x_columns, y_columns, yerr_columns):
    if len(x_columns) == 1:
        x_columns = [x_columns[0] for _ in range(len(data_frames))]
    if len(y_columns) == 1:
        y_columns = [y_columns[0] for _ in range(len(data_frames))]
    if len(yerr_columns) == 1:
        yerr_columns = [yerr_columns[0] for _ in range(len(data_frames))]
    return x_columns, y_columns, yerr_columns


def _plot_bars(ax, i, x, y, yerr, width, labels, color):
    rects = ax.bar(x + i * width, y, width, yerr=yerr, color=color,
                   ecolor='black',
                   linewidth=0)
    ax.set_xticks(x + (i / 2) * width + width)
    ax.set_xticklabels(labels, rotation=-30)
    _autolabel(rects)


def _plot_lines(ax, i, x, y, yerr, width, labels, color):
    # ax.plot(x, y)
    # ax.scatter(x, y)
    ax.errorbar(x, y, yerr=yerr, linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)


def _autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                 '%.2f' % height,
                 ha='center', va='bottom', size=5, rotation=90)


def _do_chart(data_frames, width, x_columns, y_columns,
              yerr_columns, cv=25):
    """
    parameters:
    data_frames- where to extra information from
    width- how wide the bard should be in the plot
    x,y,yerr_columns- which columns in the data frames that x,y,
    yerr data is stored. data_frames[0][x_columns[0]] contains the x value
    for the first plot, etc. It is assumed yerr contains standard deviations
    cv- how many crossvalidations were y and yerr obtained over? we divide by
     sqrt(cv) to get std error from stdev

    """

    x_columns, y_columns, yerr_columns = _safe_column_names(data_frames,
                                                            x_columns,
                                                            y_columns,
                                                            yerr_columns)

    assert len(data_frames) == len(x_columns) == len(y_columns) == \
           len(yerr_columns)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cm = plt.get_cmap('flag')
    num_groups = len(data_frames)

    for (i, (name, df)) in enumerate(data_frames):
        color = cm(1. * i / num_groups)
        x = df['sample_size']
        y = df[y_columns[i]]
        yerr = df[yerr_columns[i]] / sqrt(cv)
        labels = tuple(df[x_columns[i]])
        # _plot_bars(ax,i, x, y, yerr, width, labels, color)
        _plot_lines(ax, i, x, y, yerr, width, labels, color)
    return ax


def performance_bar_chart(prefix_tables, classifiers, width=0.1, cv=25, wheres=[]):
    prefix, tables = prefix_tables
    x_columns = ['sample_size']
    y_columns = ['score_mean']
    yerr_columns = ['score_std']

    data_frames = []
    for table in tables:
        for classifier in classifiers:
            values = {
                'classifier': classifier,
                'number': table,
                'wheres': wheres
            }
            sql = Template(perf_sql).render(values)

            print sql
            df = query_to_data_frame(sql)
            data_frames.append(('%.2d-%s' % (table, classifier),
                                df))

    ax = _do_chart(data_frames, width, x_columns, y_columns,
                   yerr_columns, cv)

    ax.set_xlabel('Sample size')
    ax.set_ylabel('Macroavg F1')
    ax.set_title('Classifier performance')

    def get_descr(frames):
        name = frames[0]
        global table_descr
        number, clf = name.split('-')
        try:
            return '{}({})-{}'.format(number, table_descr[int(number)], clf)
        except KeyError:
            return name

    ax.legend(map(get_descr, data_frames), 'lower right', prop={'size': 6})
    ax.set_ylim([0., 1.])
    exp_range = '-'.join(map(str, tables))
    classifiers = '-'.join(classifiers)
    directory = get_hash_and_date(tables[0])
    directory = os.path.join(directory, exp_range)
    if not os.path.exists(directory):
        os.mkdir(directory)
    plt.savefig('%s/%s%s-%s-perf.png' % (
        directory,
        prefix,
        classifiers,
        '_'.join(wheres)),
                format='png',
                dpi=300)


def coverage_bar_chart(experiments, width=0.13, cv=25,
                       x_columns=['sample_size'],
                       wheres=[],
                       legend_position='best',
                       xlabel='Sample size'):
    data_frames = []
    stats = [
        ['iv_it_tok_mean', 'iv_it_tok_std'],
        ['iv_oot_tok_mean', 'iv_oot_tok_std'],
        ['oov_it_tok_mean', 'oov_it_tok_std'],
        ['oov_oot_tok_mean', 'oov_oot_tok_std']
    ]

    for experiment in experiments:
        for stat in stats:
            values = {
                'variables': stat,
                'number': experiment,
                'wheres': wheres
            }
            sql = Template(coverage_sql).render(values)
            print sql
            df = query_to_data_frame(sql)
            # normalise coverage stats by total types/tokens
            df[stat[0]] = df[stat[0]] / \
                          map(float, df['total_%s' % stat[0][-8:-5]])
            df[stat[1]] = df[stat[1]] / \
                          map(float, df['total_%s' % stat[1][-7:-4]])
            name = '%.2d-%s' % (experiment, stat[0])
            data_frames.append((name, df))
    y_columns = [x[0] for x in stats]
    yerr_columns = [x[1] for x in stats]

    ax = _do_chart(data_frames, width, x_columns, y_columns,
                   yerr_columns, cv=cv)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Proportion of total tokens/types')
    ax.set_title('Thesaurus coverage')
    ax.legend(y_columns, legend_position, ncol=len(y_columns),
              prop={'size': 6})

    directory = get_hash_and_date(experiments[0])
    plt.savefig('{}/exp{}-{}-coverage.png'.format(
        directory,
        experiment,
        '_'.join(wheres)),
                format='png',
                dpi=300)


def get_hash_and_date(exp_no):
    """
    Fetches information about the git commit used to produce these
    experimental results and the date they were put into the database,
    creates and returns a directory in ./figures/ corresponding to that
    information
    """
    c = get_susx_mysql_conn().cursor()
    c.execute('SELECT DISTINCT consolidation_date from data%d' % exp_no)
    res = c.fetchall()
    date = res[0][0]

    c.execute('SELECT DISTINCT git_hash from data%d' % exp_no)
    res = c.fetchall()
    hash = res[0][0]

    directory = 'figures/{}-{}'.format(date.date(), hash[:10])
    try:
        os.mkdir(directory)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(directory):
            pass
        else:
            raise

    return directory


def myrange(start, stop_inclusive, step=1):
    return range(start, stop_inclusive + 1, step=step)

# performance_bar_chart([7, 8], ['LinearSVC'], cv=5)
# performance_bar_chart([9, 10, 11], ['MultinomialNB', 'BernoulliNB', 'LogisticRegression'], width=0.13)
# performance_bar_chart([2, 5, 6], ['LinearSVC'], cv=5)
# performance_bar_chart([9,10,11], ['MultinomialNB'])
# performance_bar_chart([9, 10, 11], ['LogisticRegression'])
# performance_bar_chart([9, 10, 11], ['BernoulliNB'])
# performance_bar_chart([12, 13, 14], ['LogisticRegression'])
# performance_bar_chart([12, 13, 14], ['BernoulliNB'])
# performance_bar_chart([12, 13, 14], ['MultinomialNB'])
# performance_bar_chart([9, 12], ['LogisticRegression'])
# performance_bar_chart([9, 12], ['BernoulliNB'])
# performance_bar_chart([9, 12], ['MultinomialNB'])
# performance_bar_chart([10, 13], ['LogisticRegression'])
table_descr = {
    22: 'IT tokens, signifier, giga',
    23: 'all tokens, signifier, giga',
    24: 'IT tokens, signifier + signified, giga',
    25: 'all token, signifier + signified, giga',
    26: 'IT tokens, signified, giga',
    27: 'all tokens, signified, giga',
    28: 'all tokens, signified random baseline',
    29: 'IT tokens, signifier, wiki',
    30: 'IT tokens, signified, wiki',
    31: 'IT tokens, signifier + signified, wiki',
    32: 'all tokens, signifier, wiki',
    33: 'all tokens, signified, wiki',
    34: 'all tokens, signifier + signified, wiki',

    35: 'AN_NN_Add, signified, gigaw',
    36: 'AN_NN_Mult, signified, gigaw',
    37: 'AN_NN_First, signified, gigaw',
    38: 'AN_NN_Second, signified, gigaw',
    39: 'AN_NN_Min, signified, gigaw',
    40: 'AN_NN_Max, signified, gigaw',
    41: 'AN_NN_Observed, signified, gigaw',

    # auto-generated by composed_thesauri_experiments.py
    42: 'AN_NN, Add, signified, gigaw-0, R2',
    43: 'AN_NN, Mult, signified, gigaw-0, R2',
    44: 'AN_NN, Left, signified, gigaw-0, R2',
    45: 'AN_NN, Right, signified, gigaw-0, R2',
    46: 'AN_NN, Min, signified, gigaw-0, R2',
    47: 'AN_NN, Max, signified, gigaw-0, R2',
    48: 'AN_NN, Observed, signified, gigaw-0, R2',

    49: 'AN_NN, Add, signified, wiki-0, R2',
    50: 'AN_NN, Mult, signified, wiki-0, R2',
    51: 'AN_NN, Left, signified, wiki-0, R2',
    52: 'AN_NN, Right, signified, wiki-0, R2',
    53: 'AN_NN, Min, signified, wiki-0, R2',
    54: 'AN_NN, Max, signified, wiki-0, R2',
    55: 'AN_NN, Observed, signified, wiki-0, R2',

    56: 'AN_NN, Add, signified, gigaw-0, MR',
    57: 'AN_NN, Mult, signified, gigaw-0, MR',
    58: 'AN_NN, Left, signified, gigaw-0, MR',
    59: 'AN_NN, Right, signified, gigaw-0, MR',
    60: 'AN_NN, Min, signified, gigaw-0, MR',
    61: 'AN_NN, Max, signified, gigaw-0, MR',
    62: 'AN_NN, Observed, signified, gigaw-0, MR',

    63: 'AN_NN, Add, signified, wiki-0, MR',
    64: 'AN_NN, Mult, signified, wiki-0, MR',
    65: 'AN_NN, Left, signified, wiki-0, MR',
    66: 'AN_NN, Right, signified, wiki-0, MR',
    67: 'AN_NN, Min, signified, wiki-0, MR',
    68: 'AN_NN, Max, signified, wiki-0, MR',
    69: 'AN_NN, Observed, signified, wiki-0, MR',

    70: 'AN_NN, Add, signified, gigaw-0, R2',
    71: 'AN_NN, Mult, signified, gigaw-0, R2',
    72: 'AN_NN, Left, signified, gigaw-0, R2',
    73: 'AN_NN, Right, signified, gigaw-0, R2',
    74: 'AN_NN, Min, signified, gigaw-0, R2',
    75: 'AN_NN, Max, signified, gigaw-0, R2',
    76: 'AN_NN, Observed, signified, gigaw-0, R2',

    77: 'AN_NN, Add, signified, gigaw-0, R2',
    78: 'AN_NN, Mult, signified, gigaw-0, R2',
    79: 'AN_NN, Left, signified, gigaw-0, R2',
    80: 'AN_NN, Right, signified, gigaw-0, R2',
    81: 'AN_NN, Min, signified, gigaw-0, R2',
    82: 'AN_NN, Max, signified, gigaw-0, R2',
    83: 'AN_NN, Observed, signified, gigaw-0, R2',

    84: 'AN_NN, Add, signified, gigaw-30, R2',
    85: 'AN_NN, Mult, signified, gigaw-30, R2',
    86: 'AN_NN, Left, signified, gigaw-30, R2',
    87: 'AN_NN, Right, signified, gigaw-30, R2',
    88: 'AN_NN, Min, signified, gigaw-30, R2',
    89: 'AN_NN, Max, signified, gigaw-30, R2',
    90: 'AN_NN, Baroni, signified, gigaw-30, R2',
    91: 'AN_NN, Observed, signified, gigaw-30, R2',

    92: 'AN_NN, Add, signified, wiki-30, R2',
    93: 'AN_NN, Mult, signified, wiki-30, R2',
    94: 'AN_NN, Left, signified, wiki-30, R2',
    95: 'AN_NN, Right, signified, wiki-30, R2',
    96: 'AN_NN, Min, signified, wiki-30, R2',
    97: 'AN_NN, Max, signified, wiki-30, R2',
    98: 'AN_NN, Baroni, signified, wiki-30, R2',
    99: 'AN_NN, Observed, signified, wiki-30, R2',

    100: 'AN_NN, Add, signified, gigaw-300, R2',
    101: 'AN_NN, Mult, signified, gigaw-300, R2',
    102: 'AN_NN, Left, signified, gigaw-300, R2',
    103: 'AN_NN, Right, signified, gigaw-300, R2',
    104: 'AN_NN, Min, signified, gigaw-300, R2',
    105: 'AN_NN, Max, signified, gigaw-300, R2',
    106: 'AN_NN, Baroni, signified, gigaw-300, R2',
    107: 'AN_NN, Observed, signified, gigaw-300, R2',

    108: 'AN_NN, Add, signified, wiki-300, R2',
    109: 'AN_NN, Mult, signified, wiki-300, R2',
    110: 'AN_NN, Left, signified, wiki-300, R2',
    111: 'AN_NN, Right, signified, wiki-300, R2',
    112: 'AN_NN, Min, signified, wiki-300, R2',
    113: 'AN_NN, Max, signified, wiki-300, R2',
    114: 'AN_NN, Baroni, signified, wiki-300, R2',
    115: 'AN_NN, Observed, signified, wiki-300, R2',

    116: 'AN_NN, Add, signified, gigaw-1000, R2',
    117: 'AN_NN, Mult, signified, gigaw-1000, R2',
    118: 'AN_NN, Left, signified, gigaw-1000, R2',
    119: 'AN_NN, Right, signified, gigaw-1000, R2',
    120: 'AN_NN, Min, signified, gigaw-1000, R2',
    121: 'AN_NN, Max, signified, gigaw-1000, R2',
    122: 'AN_NN, Baroni, signified, gigaw-1000, R2',
    123: 'AN_NN, Observed, signified, gigaw-1000, R2',

    124: 'AN_NN, Add, signified, wiki-1000, R2',
    125: 'AN_NN, Mult, signified, wiki-1000, R2',
    126: 'AN_NN, Left, signified, wiki-1000, R2',
    127: 'AN_NN, Right, signified, wiki-1000, R2',
    128: 'AN_NN, Min, signified, wiki-1000, R2',
    129: 'AN_NN, Max, signified, wiki-1000, R2',
    130: 'AN_NN, Baroni, signified, wiki-1000, R2',
    131: 'AN_NN, Observed, signified, wiki-1000, R2',

    132: 'AN_NN, Add, signified, gigaw-30, MR',
    133: 'AN_NN, Mult, signified, gigaw-30, MR',
    134: 'AN_NN, Left, signified, gigaw-30, MR',
    135: 'AN_NN, Right, signified, gigaw-30, MR',
    136: 'AN_NN, Min, signified, gigaw-30, MR',
    137: 'AN_NN, Max, signified, gigaw-30, MR',
    138: 'AN_NN, Baroni, signified, gigaw-30, MR',
    139: 'AN_NN, Observed, signified, gigaw-30, MR',

    140: 'AN_NN, Add, signified, wiki-30, MR',
    141: 'AN_NN, Mult, signified, wiki-30, MR',
    142: 'AN_NN, Left, signified, wiki-30, MR',
    143: 'AN_NN, Right, signified, wiki-30, MR',
    144: 'AN_NN, Min, signified, wiki-30, MR',
    145: 'AN_NN, Max, signified, wiki-30, MR',
    146: 'AN_NN, Baroni, signified, wiki-30, MR',
    147: 'AN_NN, Observed, signified, wiki-30, MR',

    148: 'AN_NN, Add, signified, gigaw-300, MR',
    149: 'AN_NN, Mult, signified, gigaw-300, MR',
    150: 'AN_NN, Left, signified, gigaw-300, MR',
    151: 'AN_NN, Right, signified, gigaw-300, MR',
    152: 'AN_NN, Min, signified, gigaw-300, MR',
    153: 'AN_NN, Max, signified, gigaw-300, MR',
    154: 'AN_NN, Baroni, signified, gigaw-300, MR',
    155: 'AN_NN, Observed, signified, gigaw-300, MR',

    156: 'AN_NN, Add, signified, wiki-300, MR',
    157: 'AN_NN, Mult, signified, wiki-300, MR',
    158: 'AN_NN, Left, signified, wiki-300, MR',
    159: 'AN_NN, Right, signified, wiki-300, MR',
    160: 'AN_NN, Min, signified, wiki-300, MR',
    161: 'AN_NN, Max, signified, wiki-300, MR',
    162: 'AN_NN, Baroni, signified, wiki-300, MR',
    163: 'AN_NN, Observed, signified, wiki-300, MR',

    164: 'AN_NN, Add, signified, gigaw-1000, MR',
    165: 'AN_NN, Mult, signified, gigaw-1000, MR',
    166: 'AN_NN, Left, signified, gigaw-1000, MR',
    167: 'AN_NN, Right, signified, gigaw-1000, MR',
    168: 'AN_NN, Min, signified, gigaw-1000, MR',
    169: 'AN_NN, Max, signified, gigaw-1000, MR',
    170: 'AN_NN, Baroni, signified, gigaw-1000, MR',
    171: 'AN_NN, Observed, signified, gigaw-1000, MR',

    172: 'AN_NN, Add, signified, wiki-1000, MR',
    173: 'AN_NN, Mult, signified, wiki-1000, MR',
    174: 'AN_NN, Left, signified, wiki-1000, MR',
    175: 'AN_NN, Right, signified, wiki-1000, MR',
    176: 'AN_NN, Min, signified, wiki-1000, MR',
    177: 'AN_NN, Max, signified, wiki-1000, MR',
    178: 'AN_NN, Baroni, signified, wiki-1000, MR',
    179: 'AN_NN, Observed, signified, wiki-1000, MR',
}
# performance_bar_chart([10, 13], ['BernoulliNB'])
# performance_bar_chart([10, 13], ['MultinomialNB'])
# performance_bar_chart([11, 14], ['MultinomialNB'])
# performance_bar_chart([11, 14], ['BernoulliNB'])
# performance_bar_chart([11, 14], ['LogisticRegression'])
# performance_bar_chart([17, 18, 19], ['LogisticRegression'])
# performance_bar_chart([2, 17, 18, 19], ['MultinomialNB'])

# performance_bar_chart([17, 18, 19], ['BernoulliNB'])
# performance_bar_chart([17],
#                       ['BernoulliNB', 'MultinomialNB', 'LogisticRegression'])
# performance_bar_chart([17, 18], ['BernoulliNB'])
# performance_bar_chart([17, 18], ['MultinomialNB'])
# performance_bar_chart([17, 18], ['LogisticRegression'])

# coverage_bar_chart([6, 8], cv=5, legend_position='upper center')


from joblib import Parallel, delayed

# Parallel(n_jobs=1)(delayed(coverage_bar_chart)([i], x_columns=['sample_size'])
#                    for i in range(22, 29))

# for i in range(22, 29):
#     coverage_bar_chart([i], x_columns=['sample_size'])

classifiers = ['BernoulliNB', 'MultinomialNB', 'LogisticRegression']#, 'MultinomialNBWithBinaryFeatures']
experiment_sets = [
    # [22, 23, 29], # effect of gigaw/wiki as feature selection (signifier, IT tokens) vs 23 as baseline

    # [27, 33], # giga vs wiki as feature expansion (signified, all tokens)
    #
    # [22, 24, 26], # gigaword with difference token handlers, IT tokens
    # [23, 25, 27, 28], # gigaword with difference token handlers, all tokens + random
    #
    # [29, 30, 31], # wiki with difference token handlers, IT tokens
    # # [32, 33, 34], # wiki with difference token handlers, all tokens
    #
    # [26, 27], # must be the same
    # [23, 32], # must be the same
    # range(35, 42), # all composed NP+unigram thesauri at once

    ('no_svd_', myrange(42, 48)), # all composed AN+NN thesauri, giga, R2, no SVD
    ('no_svd_', myrange(49, 55)), # all composed AN+NN thesauri, wiki, R2, no SVD
    ('no_svd_', myrange(56, 62)), # all composed AN+NN thesauri, giga, MR, no SVD
    ('no_svd_', myrange(63, 69)), # all composed AN+NN thesauri, wiki, MR, no SVD
    ('no_svd_', myrange(70, 76)), # composed AN thesauri, giga, R2
    ('no_svd_', myrange(77, 83)), # composed NN thesauri, giga, R2

    # AN+NN vs AN vs NN on gigaw, R2
    ('AN_vs_NN_', [42, 70, 77]),
    ('AN_vs_NN_', [43, 71, 78]),
    ('AN_vs_NN_', [44, 72, 79]),
    ('AN_vs_NN_', [45, 73, 80]),
    ('AN_vs_NN_', [46, 74, 81]),
    ('AN_vs_NN_', [47, 75, 82]),
    ('AN_vs_NN_', [48, 76, 83]),

    # all composed AN+NN thesauri with different SVD setting and labelled corpora
    # all thesauri from a given unlabelled corpus, one classifier, one labelled corpus
    ('svd30', myrange(84, 91)),
    ('svd30', myrange(92, 99)),
    ('svd30', myrange(132, 139)),
    ('svd30', myrange(140, 147)),
    ('svd300', myrange(100, 107)),
    ('svd300', myrange(108, 115)),
    ('svd300', myrange(148, 155)),
    ('svd300', myrange(156, 163)),

    ('svd1000', myrange(116, 123)),
    ('svd1000', myrange(124, 131)),
    ('svd1000', myrange(164, 171)),
    ('svd1000', myrange(172, 179)),

    # a different view of the above- effect of SVD
    # one thesaurus, all SVD settings, both labelled corpora

    # wiki thesauri
    ('all_svd_settings_wiki', [49, 92, 108, 124, 63, 140, 156, 172]),
    ('all_svd_settings_wiki', [50, 93, 109, 125, 64, 141, 157, 173]),
    ('all_svd_settings_wiki', [51, 94, 110, 126, 65, 142, 158, 174]),
    ('all_svd_settings_wiki', [52, 95, 111, 127, 66, 143, 159, 175]),
    ('all_svd_settings_wiki', [53, 96, 112, 128, 67, 144, 160, 176]),
    ('all_svd_settings_wiki', [54, 97, 113, 129, 68, 145, 161, 177]),
    ('all_svd_settings_wiki', [55, 99, 115, 131, 69, 147, 163, 179]), #observed
    ('all_svd_settings_wiki', [98, 114, 130, 146, 162, 178]), #baroni


    # gigaw thesauri
    ('all_svd_settings_giga', [42, 84, 100, 116, 56, 132, 148, 164]), # add, mult, left, right, min, max
    ('all_svd_settings_giga', [43, 85, 101, 117, 57, 133, 149, 165]),
    ('all_svd_settings_giga', [44, 86, 102, 118, 58, 134, 150, 166]),
    ('all_svd_settings_giga', [45, 87, 103, 119, 59, 135, 151, 167]),
    ('all_svd_settings_giga', [46, 88, 104, 120, 60, 136, 152, 168]),
    ('all_svd_settings_giga', [47, 89, 105, 121, 61, 137, 153, 169]),
    ('all_svd_settings_giga', [48, 91, 107, 123, 62, 139, 155, 171]), #observed
    ('all_svd_settings_giga', [90, 106, 122, 138, 154, 170]), #baroni
]

Parallel(n_jobs=4)(delayed(performance_bar_chart)(experiments, [clf])
                   for clf in classifiers for experiments in experiment_sets)
# for clf in classifiers:
#     for experiments in experiments:
#         performance_bar_chart(experiments, [clf])

print 'done'
