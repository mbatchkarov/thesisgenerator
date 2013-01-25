'''
Created on Oct 23, 2012

@author: ml249
'''

import os
import re
import csv
import numpy as np
import glob
from matplotlib import pyplot as plt
import functools

import statistics


params = {'font.size': 8,
          'axes.titlesize': 8,
          'lines.markersize': 2,
          'lines.linewidth': 0.6,
          'figure.subplot.top': 0.95,
          'figure.subplot.bottom': 0.15,
          'figure.subplot.left': 0.08,
          'figure.subplot.right': 0.97,
          'figure.figsize': (8, 6.0),
          'figure.subplot.hspace': 0.25,
          'figure.dpi': 120,
          'axes.linewidth': 0.3,
          'grid.linewidth': 0.1,
          'xtick.major.size': 2,
          'ytick.major.size': 2,
          'axes.grid': True}
plt.rcParams.update(params)
#                  'subplot.wspace':,
#                  'axes.labelsize': 8,
#                  'text.fontsize': 8,
#                  'legend.fontsize': 11,
#                  'xtick.labelsize': 11,
#                  'ytick.labelsize': 11,

# todo: I need to make the csv reader return a dictionary

def get_axis_values(column_id, lines, csv_header=['tp', 'fp', 'tn', 'fn']):
    """Returns a list of specific column values from each line of a csv file.
    
    The *column_id* should be either the column name from a csv file or one of
    the statistics in the ``thesis_generator.statistics`` module.
    
    *lines* should be a list of lines from a csv file. Each line should be a
    dictionary containing at a minimum a confusion matrix. If the lines are not
    dictionaries they will be converted into dictionaries using *csv_header* as
    the keys for each line.
    
    *csv_header* should be the header line from the csv file passed in to the
    function.
    
    If the column_id is a statistic each line from the csv file is passed to the
    statistic function in turn as a dictionary. 
    """
    if type(lines[0]) != 'dict':
        lines = [{k: float(v) for k, v in zip(csv_header, line)} for line in
                 lines]

    # Construct the scoring function from the column_id defined on the command
    # line. The scoring function is a partial function that takes as argument
    # a line from the csv file and then calls the source function with the line
    # as an argument
    column_id = column_id.strip().lower()
    try:
        stat = functools.partial(getattr(statistics, column_id))
    except AttributeError:
        stat = functools.partial(lambda line: line[column_id])

    # get the specified values from the csv lines
    values = map(stat, lines)
    return values


def execute(args):
    groups_re = re.compile('([\w]+)\.')
    files = []
    #    for fname in os.listdir(args.create_figures):
    for fpath in glob.glob(os.path.join(args.create_figures, '*')):
    #        path = os.path.join(args.create_figures, fname)
        tup = tuple([fpath] + groups_re.findall(fpath))
        files.append(tup)

    if not os.path.exists(os.path.join(args.output, 'figures')):
        os.makedirs(os.path.join(args.output, 'figures'))

    # get the values of the settings according to which the files should be
    # grouped together
    plots = set([fname[args.figure_grouping] for fname in files])

    # based on the values of the setting group the files 
    plot_groups = [
    [fname for fname in files if fname[args.figure_grouping] == key]\
    for key in plots]

    for plot_group in plot_groups:
        if args.line_grouping is None:
            line_groups = [[group] for group in plot_group]
        else:
            lines = set([fname[args.line_grouping] for fname in files])
            line_groups = [[group for group in plot_group if\
                            group[args.line_grouping] == key] for key in lines]
        #        print line_groups
        fig = plt.figure()
        axes = plt.subplot(111)
        plt.hold(b=True)

        for line_group in line_groups:
            fh = open(line_group[0][0], 'r')
            num_lines = len(fh.read().strip().split('\n')) - 1
            fh.close()
            values = np.ndarray(shape=(len(line_group), 2, num_lines),
                dtype=np.float64)

            for i, item in enumerate(line_group):
                # todo: average over the line group
                with open(item[0], 'r') as in_fh:
                    reader = csv.reader(in_fh)
                    csv_header = [field.strip().lower() for field in
                                  reader.next()]

                    lines = [line for line in reader]

                    x_values = get_axis_values(args.x_values, lines,
                        csv_header)
                    y_values = get_axis_values(args.y_values, lines,
                        csv_header)
                    values[i] = [x_values, y_values]

                #            print item
            plt.plot(np.mean(values[:, 0, :], axis=0),
                np.mean(values[:, 1, :], axis=0),
                label=' '.join(map(lambda i: item[i], args.line_label)))

        plt.xlabel(args.x_values)
        plt.ylabel(args.y_values)
        plt.title(' '.join(map(lambda i: item[i], args.figure_label)))
        leg = plt.legend(loc=( 0.0, -0.165 ), ncol=args.legend_ncol,
            fancybox=True, shadow=True,
            prop={'size': 9}, mode='expand')
        leg.get_frame().set_lw(0.3)
        axes.minorticks_on()
        figure_fn = os.path.join(args.output, 'figures',
            '%s.png' % ('.'.join(plot_group[0][1:])))
        plt.savefig(figure_fn)

        print '--> save figure to: \'%s\'' % figure_fn


