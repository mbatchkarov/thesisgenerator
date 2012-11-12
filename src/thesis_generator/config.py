'''
Created on Oct 27, 2012

@author: mattilyra
'''

import argparse
import textwrap

# **********************************
# SETUP ARGUMENTS PARSER
# **********************************

arg_parser = argparse.ArgumentParser(usage='%(prog)s SOURCE_CORPUS '\
                                           'OUTPUT_DIR CONF_FILE [-cp CLASSPATH]'\
                                           '[--clean][--crossvalidate METHOD][-k]',
    formatter_class=argparse.RawTextHelpFormatter,
    description=textwrap.dedent('''\
                     Classifier experimentation framework
                     -----------------------------------------------------------
                         The classifier experimentation framework is a wrapper
                         to classifiers from libsvm, liblinear and mallet. It
                         allows easy access to training and testing the
                         classifiers and provides evaluation methods.
                                         
                         The basic usage of the framework is to define a source
                         file and an output directory, a list of actions to be
                         performed by the framework and options for those
                         actions.
                     -----------------------------------------------------------
                     -----------------------------------------------------------
                                     '''))

arg_parser.add_argument('source',
    help=textwrap.dedent('''\
                        Input file or directory for raw data when performing the
                        data split. When performing feature selection or
                        classifier training evaluation the file name(s) in this
                        variable are used to find the intermediary that contain
                        train/test data.'''),
    type=str,
    metavar='INPUT_DIR/FILE')

arg_parser.add_argument('output',
    help=textwrap.dedent('''\
                        Output directory. All the output from a group a group of
                        experiments will be written to the specified directory.
                        A predetermined file and directory hiearchy will be
                        created.'''),
    type=str,
    metavar='OUTPUT_DIR')

arg_parser.add_argument('configuration',
    help=textwrap.dedent('''\
                        A configuration file to read the processing steps that
                        should be performed by the framework and settings for
                        those steps.'''),
    type=str,
    metavar='CONF_FILE')

arg_parser.add_argument('--clean', help=textwrap.dedent('''\
                    Clean the output directory of all files before running any
                    other commands.'''),
    action='store_true', default=False)

arg_parser.add_argument('--version', action='version', version='%(prog)s 0.2')

arg_parser.add_argument('-cp', '--classpath',
    help=textwrap.dedent('''\
                        Classpath for searching binaries. If several directories
                        are provided they should be separated using the system
                        path separator (\':\' on *nix)'''),
    default='.')

arg_parser.add_argument('--log-path',
    help=textwrap.dedent('''Output directory for log files.'''),
    default='./logs')

