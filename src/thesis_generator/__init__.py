'''
Created on Oct 18, 2012

@author: ml249
'''

import argparse


# **********************************
# SETUP ARGUMENTS PARSER
# **********************************

arg_parser = argparse.ArgumentParser(description='Launch an experiment.')

action_group = arg_parser.add_argument_group()
action_group.add_argument('--split-data', help='Split a data file into \
                training and test data.', action='store_true')
action_group.add_argument('--feature-selection', help='Perform feature \
                selection on input train and test files.', action='store_true')
action_group.add_argument('--train', help='Train a model. To train a model\
                the --feature-selector and --feature-count have to defined\
                so that the correct input files can be read into the training.',
                action='store_true', default=False)
action_group.add_argument('--predict', help='Evaluate a model',
                          action='store_true', default=False)
action_group.add_argument('--clean', help='Clean the output directory of all \
                files before running any other commands',
                action='store_true', default=False)
action_group.add_argument('--create-tables', help='Create confusion matrices \
                across all the specified experimental conditions.',
                type=str, default=None, metavar='CLASSIFICATIONS_DIR')
action_group.add_argument('--create-figures', help='Create plots from \
                confusion matrix tables',
                type=str, default=None, metavar='TABLES_DIR')

arg_parser.add_argument('-id', '--jobid',
                        help='A numerical id for the job. This is used as a \
                            prefix to output files.',
                        type=int,
                        default=1,
                        metavar='1')

arg_parser.add_argument('-o', '--output',
                        help='Output directory.',
                        type=str,
                        required=True,
                        metavar='OUTPUT_DIR')

arg_parser.add_argument('-s', '--source',
                        help='Input file or directory for raw data when \
                        performing the data split. When performing feature \
                        selection or classifier training evaluation the file \
                        name(s) in this variable are used to find the \
                        intermediary that contain train/test data.',
                        type=str,
                        metavar='INPUT_DIR/FILE',
                        required=False)

arg_parser.add_argument('-cp', '--classpath',
                        help='Classpath for searching binaries. If several \
                        directories are provided they should be separated \
                        using the system path separator (\':\' on *nix)',
                        default='.')

split_data_group = arg_parser.add_argument_group('Splitting data')
split_data_group.add_argument('--stratify',
                        help='Stratify training data.',
                        action='store_true',
                        default=False)

split_data_group.add_argument('--seen-data-cutoff',
                              help='How many positive articles should be \
                                  considered to be the seen data from which \
                                  the training data is sampled.',
                              type=int,
                              default=200,
                              metavar='200')

split_data_group.add_argument('--train-data-size',
                        help='The number of positive documents to add to the \
                            training data.',
                        type=int,
                        default=200,
                        metavar='200')

feature_selection_group = arg_parser.add_argument_group('Performing feature \
                                                        selection')
feature_selection_group.add_argument('-sm', '--scoring-metric',
                        help='Feature selection scoring metric to be used to \
                            order the features found in the training data and \
                            prune the training and testing files. Test files \
                            are limited to the features seen in the training \
                            data.',
                        type=str,
                        choices=['rand','acc','acc2','bns','chi2','dfreq',
                                 'f1','ig','oddn','odds','pr','pow_k'],
                        default=[],
                        nargs='+')

feature_selection_group.add_argument('-fc', '--feature-count',
                        help='Number of features to be selected from the \
                            training data.',
                        type=int,
                        default=None)

train_group = arg_parser.add_argument_group('Training and evaluating classifier(s)')
train_group.add_argument('-c', '--classifiers',
                        help='Which classifier(s) should be trained and tested.',
                        type=str,
                        nargs='+',
                        choices=['libsvm','liblinear',
                                 'mallet_maxent','mallet_nb'])

train_group.add_argument('--crossvalidate',
                        help='Perform crossvalidation.',
                        action='store_true')

train_group.add_argument('--prc-args',
                        help='A space separated string of arguments to be \
                            passed to the subprocess running the classifiers. \
                            At a minimum the location of the target executable \
                            has to be specified. Please refer to the \
                            documentation of the respective libraries used to \
                            classify documents to see how the library can be \
                            configured.',
                        type=str,
                        default='')

plot_group = arg_parser.add_argument_group('Plotting figures')
plot_group.add_argument('--figure-grouping',
                        help='',
                        default=0,
                        type=int)

plot_group.add_argument('--line-grouping',
                        help='',
                        default=0,
                        type=int)

plot_group.add_argument('--line-label',
                        help='The position of the setting value to be used as\
                        the line label.',
                        type=int,
                        default=[1],
                        nargs='+')

plot_group.add_argument('--legend-ncol',
                        help='The number of columns to use in the legend',
                        type=int,
                        default=2,
                        metavar='2')

plot_group.add_argument('--x-values',
                        help='',
                        default='specificity',
                        type=str)

plot_group.add_argument('--y-values',
                        help='',
                        default='recall',
                        type=str)
