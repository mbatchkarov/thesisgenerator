# coding=utf-8
import ast
import csv
import glob
import os
import re
import datetime
import git
import numpy
from thesisgenerator.utils.conf_file_utils import parse_config_file


"""
Goes through output and log files for a given experiment and collects
interesting information. It is then passed on to a writer object,
which may write it to a csv or to a database
"""


def consolidate_results(writer, conf_dir, output_dir):
    """
    Consolidates the results of a series of experiment and passes it on to a
    writer
    A single thesaurus must be used in each experiment
    """
    print('Consolidating results from %s' % conf_dir)

    experiments = glob.glob(os.path.join(conf_dir, '*.conf'))
    for conf_file in experiments:
        print('Processing file %s' % conf_file)

        config_obj, configspec_file = parse_config_file(conf_file)

        exp_name = config_obj['name']
        cv_folds = config_obj['crossvalidation']['k']
        sample_size = config_obj['crossvalidation']['sample_size']

        # find out the classifier score from the final csv file
        output_file = os.path.join(output_dir, '%s.out.csv' % exp_name)
        git_hash = git.Repo('.').head.commit.hexsha[:10]

        try:
            reader = csv.reader(open(output_file, 'r'))
            _ = reader.next()  # skip over header
            for row in reader:
                classifier, metric, score_my_mean, score_my_std = row

                writer.writerow([
                    None,  # primary key, should be updated automatically
                    exp_name,
                    git_hash,
                    datetime.datetime.now().isoformat(),

                    # experiment settings
                    cv_folds,
                    sample_size,  #sample_size
                    classifier,
                    # these need to be converted to a bool and then to an int
                    #  because mysql stores booleans as a tinyint and complains
                    #  if you pass in a python boolean

                    # performance
                    metric,
                    score_my_mean,
                    score_my_std])
        except IOError:
            print('WARNING: %s is missing' % output_file)
            continue  # file is missing
