# coding=utf-8
from math import sqrt
import operator

import numpy
from thesisgenerator import utils

__author__ = 'mmb28'

import matplotlib.pyplot as plt
import pandas.io.sql as psql


def query_to_data_frame(sql):
    return psql.frame_query(sql, utils.get_susx_mysql_conn())


# <codecell>

def _safe_column_names(data_frames, x_columns, y_columns, yerr_columns):
    if len(x_columns) == 1:
        x_columns = [x_columns[0] for _ in range(len(data_frames))]
    if len(y_columns) == 1:
        y_columns = [y_columns[0] for _ in range(len(data_frames))]
    if len(yerr_columns) == 1:
        yerr_columns = [yerr_columns[0] for _ in range(len(data_frames))]
    return x_columns, y_columns, yerr_columns


def _grouped_bar_chart(data_frames, width, x_columns, y_columns,
                       yerr_columns, cv=25, hatch=False):
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
    cm = plt.get_cmap('gist_rainbow')
    num_groups = len(data_frames)
    # todo either predefine a set of colors of a set of hatch styles, not both
    ax.set_color_cycle([cm(1. * i / num_groups) for i in range(num_groups)])

    hatches = " " * len(data_frames)
    if hatch:
        hatches = "\\-/x*."

    for (i, (name, df)) in enumerate(data_frames):
        color = cm(1. * i / num_groups)
        x = numpy.arange(len(df[x_columns[i]]))
        y = df[y_columns[i]]
        yerr = df[yerr_columns[i]] / sqrt(cv)
        ax.bar(x + i * width, y, width, yerr=yerr, color=color, ecolor='black',
               linewidth=0, hatch=hatches[i % len(data_frames)])
        ax.set_xticks(x + (i / 2) * width + width)
        ax.set_xticklabels(tuple(df[x_columns[i]]), rotation=-30)
    return ax


def performance_bar_chart(tables, classifiers, width=0.2, cv=25):
    x_columns = ['sample_size']
    y_columns = ['score_mean']
    yerr_columns = ['score_std']

    data_frames = []
    for table in tables:
        for classifier in classifiers:
            sql = "SELECT sample_size,score_mean,score_std FROM data%.2d " \
                  "where " \
                  "metric = \"macroavg_f1\" and" \
                  " classifier = \"%s\"" % (table, classifier)
            print sql
            df = query_to_data_frame(sql)
            # print df
            data_frames.append(('%.2d-%s' % (table, classifier),
                                df))

    ax = _grouped_bar_chart(data_frames, width, x_columns, y_columns,
                            yerr_columns, cv)

    ax.set_xlabel('Sample size')
    ax.set_ylabel('Macroavg F1')
    ax.set_title('Classifier performance')
    ax.legend(map(operator.itemgetter(0), data_frames), 'best')
    ax.set_ylim([0., 1.])
    exp_range = '-'.join(map(str, tables))
    classifiers = '-'.join([x[:5] for x in classifiers])
    plt.savefig('figures/exp%s-%s-performa.png' % (exp_range, classifiers),
                format='png',
                dpi=300)


def coverage_bar_chart(experiments, width=0.13, cv=25,
                       x_columns=['sample_size'], legend_position='best'):
    data_frames = []
    stats = [
        ["unknown_tok_mean", "unknown_tok_std"],
        ["found_tok_mean", "found_tok_std"],
        ["replaced_tok_mean", "replaced_tok_std"],
        ["unknown_typ_mean", "unknown_typ_std"],
        ["found_typ_mean", "found_typ_std"],
        ["replaced_typ_mean", "replaced_typ_std"]
    ]

    for experiment in experiments:
        for stat in stats:
            sql = "SELECT DISTINCT name, sample_size,total_tok,total_typ,%s " \
                  "FROM data%.2d" % (','.join(stat), experiment)
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

    ax = _grouped_bar_chart(data_frames, width, x_columns, y_columns,
                            yerr_columns, hatch=True, cv=cv)

    # ax.set_xlabel('Sample size')
    ax.set_ylabel('Proportion of total tokens/types')
    ax.set_title('Thesaurus coverage')
    ax.legend(y_columns, legend_position, ncol=len(y_columns),
              prop={'size': 6})
    plt.savefig('figures/exp%s-coverage.png' % experiment, format='png',
                dpi=300)


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
# performance_bar_chart([10, 13], ['BernoulliNB'])
# performance_bar_chart([10, 13], ['MultinomialNB'])
# performance_bar_chart([11, 14], ['MultinomialNB'])
# performance_bar_chart([11, 14], ['BernoulliNB'])
# performance_bar_chart([11, 14], ['LogisticRegression'])
# performance_bar_chart([17, 18, 19], ['LogisticRegression'])
# performance_bar_chart([2, 17, 18, 19], ['MultinomialNB'])

# performance_bar_chart([17, 18, 19], ['BernoulliNB'])
performance_bar_chart([17],
                      ['BernoulliNB', 'MultinomialNB', 'LogisticRegression'])
performance_bar_chart([17, 18], ['BernoulliNB'])
performance_bar_chart([17, 18], ['MultinomialNB'])
performance_bar_chart([17, 18], ['LogisticRegression'])

# coverage_bar_chart([6, 8], cv=5, legend_position='upper center')
# coverage_bar_chart([16], x_columns=['name'])
print 'done'
