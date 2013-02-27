import operator

import numpy


__author__ = 'mmb28'


# -*- coding: utf-8 -*-
# <nbformat>2</nbformat>

# <codecell>
database_path = '/Volumes/LocalScratchHD/LocalHome/Dropbox/work/bov_data.sqlite'
import sqlite3 as lite
import matplotlib.pyplot as plt
import pandas.io.sql as psql


def query_to_data_frame(sql):
    #open the database
    global database_path
    con = lite.connect(database_path)
    with con:
        return psql.frame_query(sql, con)


# <codecell>

def performance_bar_chart(tables, classifiers,width = 0.13):
    data_frames = []
    for table in tables:
        for classifier in classifiers:
            sql = "SELECT * FROM data%.2d where metric == \"macroavg_f1\" and" \
                  " classifier == \"%s\"" % (table, classifier)
            print sql
            data_frames.append(('%.2d-%s' % (table, classifier),
                                query_to_data_frame(sql)))
           # the width of the bars
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cm = plt.get_cmap('gist_rainbow')
    num_groups = len(data_frames)
    # todo either predefine a set of colors of a set of hatch styles, not both
    ax.set_color_cycle([cm(1. * i / num_groups) for i in range(num_groups)])
    for (i, (name, df)) in enumerate(data_frames):
        color = cm(1. * i / num_groups)
        x = numpy.arange(len(df['sample_size']))
        y = df['score_mean']
        yerr = df['score_std']
        ax.bar(x + i * width, y, width, yerr=yerr, color=color, ecolor='black',
            linewidth=0, hatch="\\-/x*."[i % len(data_frames)])
        ax.set_xlabel('Sample size')
        ax.set_ylabel('Macroavg F1')
        # ax.set_title(
        #     'Earn vs not-earn with different thesauri and training data')
        ax.set_xticks(x + (i / 2) * width + width)
        ax.set_xticklabels(tuple(df['sample_size']))
    ax.legend(map(operator.itemgetter(0), data_frames), 'lower right')
    exp_range = '-'.join(map(str, tables))
    plt.savefig('figures/exp%s-performa.png'%exp_range, format='png',
        dpi=300)
    # plt.show()


performance_bar_chart([7,8], ['LinearSVC'])
performance_bar_chart(range(9,12), ['LinearSVC', 'BernoulliNB'])

print 'done'
