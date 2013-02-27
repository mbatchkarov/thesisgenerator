import operator

import numpy
import matplotlib


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


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
            '%d' % int(height),
            ha='center', va='bottom')


# <codecell>

data_frames = []
for table in range(9, 12):
    for classifier in ['LinearSVC', 'BernoulliNB']:
        sql = "SELECT * FROM data%.2d where metric == \"macroavg_f1\" and" \
              " classifier == \"%s\"" % (table, classifier)
        print sql
        data_frames.append(('%.2d-%s' % (table, classifier),
                            query_to_data_frame(sql)))

width = 0.1       # the width of the bars
fig = plt.figure()
ax = fig.add_subplot(111)
cm = plt.get_cmap('gist_rainbow')
NUM_COLORS = 10
ax.set_color_cycle([cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])

for (i, (name, df)) in enumerate(data_frames):
    color = cm(1. * i / NUM_COLORS)
    x = numpy.arange(len(df['sample_size']))
    y = df['score_mean']
    yerr = df['score_std']
    ax.bar(x + i * width, y, width, yerr=yerr, color=color)
    ax.set_xlabel('Sample size')
    ax.set_ylabel('Macroavg F1')
    ax.set_title('Earn vs not-earn with different thesauri and training data')
    ax.set_xticks(x + width, tuple(df['sample_size']))

ax.legend(map(operator.itemgetter(0), data_frames), 'lower right')
plt.savefig('figures/exp9-11-perf.png', format='png', dpi=fig.dpi)
plt.show()

print 'done'
