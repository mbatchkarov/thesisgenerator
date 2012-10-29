'''
Created on Oct 27, 2012

@author: mattilyra
'''

import argparse
import textwrap


# **********************************
# SETUP ARGUMENTS PARSER
# **********************************

arg_parser = argparse.ArgumentParser(usage='%(prog)s -s FILE '\
                                     '-o OUTPUT_DIR [-cp CLASSPATH] '\
                                     '[actions][options]',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description=textwrap.dedent('''\
                     Classifier experimentation framework
                     -----------------------------------------------------------
                         The classifier experimentation framework is a wrapper
                         to classifiers from libsvm, liblinear and mallet. It
                         allows easy access to training and testing the
                         classifiers and provides rudimentary evaluation methods.
                                         
                         The basic usage of the framework is to define a source
                         file and an output directory, a list of actions to be
                         performed by the framework and options for those
                         actions.
                     -----------------------------------------------------------
                     -----------------------------------------------------------
                                     '''))

action_group = arg_parser.add_argument_group()
action_group.add_argument('--split-data',
                          help=textwrap.dedent('''\
                              Split a data file into training and test data.'''),
                          action='store_true')

action_group.add_argument('--feature-selection', help='Perform feature '\
                'selection on input train and test files.', action='store_true')

action_group.add_argument('--train', help=textwrap.dedent('''\
                                Train a model. To train a model the
                                --scoring-metric and --feature-count options
                                have to defined so that the correct input files
                                can be read into the training.'''),
                          action='store_true', default=False)

action_group.add_argument('--predict', help='Evaluate a model',
                          action='store_true', default=False)

action_group.add_argument('--clean', help=textwrap.dedent('''\
                    Clean the output directory of all files before running any
                    other commands.'''),
                          action='store_true', default=False)

action_group.add_argument('--create-tables', help=textwrap.dedent('''\
                    Create confusion matrices across all the specified
                    experimental conditions.'''),
                          type=str, default=None, metavar='CLASSIFICATIONS_DIR')
action_group.add_argument('--create-figures', help=textwrap.dedent('''\
                    Create plots from confusion matrix tables.'''),
                type=str, default=None, metavar='TABLES_DIR')

arg_parser.add_argument('--version', action='version', version='%(prog)s 0.2')

arg_parser.add_argument('-id', '--jobid',
                        help=textwrap.dedent('''\
                        Numerical job id, used as a prefix to output files.
                        Currently not used.'''),
                        type=int,
                        default=1,
                        metavar='1')

arg_parser.add_argument('-o', '--output',
                        help=textwrap.dedent('''\
                        Output directory. All the output from a group a group of
                        experiments will be written to the specified directory.
                        A predetermined file and directory hiearchy will be
                        created.'''),
                        type=str,
                        required=True,
                        metavar='OUTPUT_DIR')

arg_parser.add_argument('-s', '--source',
                        help=textwrap.dedent('''\
                        Input file or directory for raw data when performing the
                        data split. When performing feature selection or
                        classifier training evaluation the file name(s) in this
                        variable are used to find the intermediary that contain
                        train/test data.'''),
                        type=str,
                        metavar='INPUT_DIR/FILE',
                        required=False)

arg_parser.add_argument('-cp', '--classpath',
                        help=textwrap.dedent('''\
                        Classpath for searching binaries. If several directories
                        are provided they should be separated using the system
                        path separator (\':\' on *nix)'''),
                        default='.')

split_data_group = arg_parser.add_argument_group('Splitting data')
split_data_group.add_argument('--stratify',
                        help='Stratify training data.',
                        action='store_true',
                        default=False)

split_data_group.add_argument('--num-seen',
                              help=textwrap.dedent('''\
                          How many positive articles should be considered to be
                          the seen data from which the training data is sampled.'''),
                              type=int,
                              default=200,
                              metavar='200')

split_data_group.add_argument('--train-data-size',
                        help=textwrap.dedent('''\
                        The number of positive documents to add to the training
                        data.'''),
                              type=int,
                              default=200,
                              metavar='200')

feature_selection_group = arg_parser.add_argument_group('Performing feature '\
                                                        'selection')
feature_selection_group.add_argument('-sm', '--scoring-metric',
                        help=textwrap.dedent('''\
                        Feature selection scoring metric to be used to order the
                        features found in the training data and prune the
                        training and testing files. Test files are pruned to
                        only contain  features seen in the training data.
                        Available selection metrics:
                            * rand (Random)
                            * acc (Accuracy)
                            * acc2 (Accuracy balanced) |tpr - fpr|
                            * bns (Binormal Separation)
                            * chi2 (Chi Squared)
                            * dfreq (Document Frequency)
                            * f1 (F1 measure)
                            * ig (Information gain)
                            * oddn (Odds ratio numerator)
                            * odds (Odds ratio)
                            * pr (Probability ratio)
                            * pow_k (1-fpr)^k - (1-tpr)^k.
                        tpr = true positive rate
                        fpr = false positive rate'''),
                        type=str,
                        choices=['rand','acc','acc2','bns','chi2','dfreq',
                                 'f1','ig','oddn','odds','pr','pow_k'],
                        default=[],
                        nargs='+',
                        metavar='chi2 ig bns')

feature_selection_group.add_argument('-fc', '--feature-count',
                        help=textwrap.dedent('''
                        Number of features to be selected from the training data.'''),
                        type=int,
                        default=None)

train_group = arg_parser.add_argument_group('Training and evaluating classifier(s)')
train_group.add_argument('-c', '--classifiers',
                        help=textwrap.dedent('''\
                        Which classifier(s) should be trained and tested.
                        
                        All of the Mallet classifiers should be prefixed by
                        "mallet_" followed by the name of classifier trainer
                        class (without "Trainer"), for instance mallet_MaxEnt.
                        
                        See the --prc-args option for passing options to the
                        classifier subprocess.
                        
                        Available classifiers:
                            * libsvm (LibSVM support vector machine)
                            * liblinear (LibLinear support vector machine) 
                            * mallet_{AdaBoost,AdaBoostM2,Bagging,BalancedWinnow,C45,DecisionTree,MaxEntGE,MaxEntGERange,MaxEntL1,MaxEntPR,MCMaxEnt,NaiveBayes,NaiveBayesEM,RankMaxEnt,Winnow}'''),
                        type=str,
                        nargs='+',
                        choices=['libsvm','liblinear',
                                 'mallet_AdaBoost','mallet_AdaBoostM2'\
                                 'mallet_Bagging','mallet_BalancedWinnow',\
                                 'mallet_C45','mallet_DecisionTree',\
                                 'mallet_MaxEntGE','mallet_MaxEntGERange',\
                                 'mallet_MaxEntL1', 'mallet_MaxEntPR',\
                                 'mallet_MCMaxEnt','mallet_NaiveBayes',\
                                 'mallet_NaiveBayesEM','mallet_RankMaxEnt',\
                                 'mallet_Winnow'],
                         metavar='')

#train_group.add_argument('--crossvalidate',
#                        help='Perform crossvalidation.',
#                        action='store_true')

train_group.add_argument('--prc-args',
                        help=textwrap.dedent('''\
                        A space separated string of arguments to be passed to
                        the subprocess running the classifiers. Please refer to
                        the documentation of the respective libraries used to
                        classify documents to see how the library can be
                        configured.
                        
                        Note: Passing arguments to the subprocesses means that
                        only a single classifier model can be trained per run.'''),
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
                        help=textwrap.dedent('''\
                        The position of the setting value to be used as the line
                        label.'''),
                        type=int,
                        default=[1],
                        nargs='+')

plot_group.add_argument('--legend-ncol',
                        help=textwrap.dedent('''\
                        The number of columns to use in the legend'''),
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
