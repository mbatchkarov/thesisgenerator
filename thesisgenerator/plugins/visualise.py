# coding=utf-8
from math import sqrt
from jinja2 import Template

import numpy
from thesisgenerator import utils

__author__ = 'mmb28'

import matplotlib.pyplot as plt
import pandas.io.sql as psql

coverage_sql = \
    "    SELECT DISTINCT name, sample_size,total_tok,total_typ, " \
    "{{ variables|join(', ') }} FROM data{{ '{:02d}'.format(number)}} " \
    "{% if wheres %} WHERE {{ wheres|join(' and ') }} {% endif %} " \
    "ORDER BY sample_size"

perf_sql = "SELECT sample_size,score_mean,score_std " \
           "FROM data{{ '{:02d}'.format(number)}} " \
           "WHERE metric=\"macroavg_f1\" and classifier=\"{{classifier}}\"" \
           "{% if wheres %} and {{ wheres|join(' and ') }} {% endif %} " \
           "ORDER BY sample_size"


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

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                     '%.2f' % height,
                     ha='center', va='bottom', size=5, rotation=90)

    x_columns, y_columns, yerr_columns = _safe_column_names(data_frames,
                                                            x_columns,
                                                            y_columns,
                                                            yerr_columns)

    assert len(data_frames) == len(x_columns) == len(y_columns) == \
           len(yerr_columns)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cm = plt.get_cmap('Accent')
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
        rects = ax.bar(x + i * width, y, width, yerr=yerr, color=color,
                       ecolor='black',
                       linewidth=0, hatch=hatches[i % len(data_frames)])
        ax.set_xticks(x + (i / 2) * width + width)
        ax.set_xticklabels(tuple(df[x_columns[i]]), rotation=-30)
        autolabel(rects)
    return ax


def performance_bar_chart(tables, classifiers, width=0.1, cv=25, wheres=[]):
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

    ax = _grouped_bar_chart(data_frames, width, x_columns, y_columns,
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
    plt.savefig('figures/exp%s-%s-%s-perf.png' % (exp_range,
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
            # sql = "SELECT DISTINCT name, sample_size,total_tok,total_typ,%s " \
            #       "FROM data%.2d order by sample_size " % (','.join(stat),
            #                                                experiment)

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

    ax = _grouped_bar_chart(data_frames, width, x_columns, y_columns,
                            yerr_columns, hatch=False, cv=cv)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Proportion of total tokens/types')
    ax.set_title('Thesaurus coverage')
    ax.legend(y_columns, legend_position, ncol=len(y_columns),
              prop={'size': 6})
    plt.savefig('figures/exp%s-%s-coverage.png' % (experiment,
                                                   '_'.join(wheres)),
                format='png',
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
# performance_bar_chart([17],
#                       ['BernoulliNB', 'MultinomialNB', 'LogisticRegression'])
# performance_bar_chart([17, 18], ['BernoulliNB'])
# performance_bar_chart([17, 18], ['MultinomialNB'])
# performance_bar_chart([17, 18], ['LogisticRegression'])

# coverage_bar_chart([6, 8], cv=5, legend_position='upper center')

table_descr = {
    22: 'IT tokens, signifier',
    23: 'all tokens, signifier',
    24: 'IT tokens, signifier + signified',
    25: 'all token, signifier + signified',
    26: 'IT tokens, signified',
    27: 'all tokens, signified'
}

for i in range(22, 28):
    coverage_bar_chart([i], x_columns=['sample_size'])

for clf in ['BernoulliNB', 'MultinomialNB',
            'LogisticRegression', 'MultinomialNBWithBinaryFeatures']:
    for experiments in [[23, 22], [25, 24], [27, 26], [22, 26, 24],
                        [23, 27, 25], range(22, 28)]:
        performance_bar_chart(experiments, [clf])

print 'done'
