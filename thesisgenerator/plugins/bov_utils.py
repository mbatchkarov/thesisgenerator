from copy import deepcopy
import csv
import glob
from itertools import chain
import os
import re
import shutil
import sys
import ast
import itertools
import numpy
from numpy import nonzero

try:
    from thesisgenerator.__main__ import go, _get_data_iterators, parse_config_file
    from thesisgenerator.utils import replace_in_file
except ImportError:
# if one tries to run this script from the main project directory the
# thesisgenerator package would not be on the path, add it and try again
    sys.path.append('../../')
    sys.path.append('../')
    sys.path.append('./')
    sys.path.append('./thesisgenerator')
    from thesisgenerator.__main__ import go, _get_data_iterators, parse_config_file
    from thesisgenerator.utils import replace_in_file

__author__ = 'mmb28'


def inspect_thesaurus_effect(outdir, clf_name, thesaurus_file, pipeline,
                             predicted, x_test):
    """
    Evaluates the performance of a classifier with and without the thesaurus
    that backs its vectorizer
    """

    # remove the thesaurus
    pipeline.named_steps['vect'].thesaurus = {}
    predicted2 = pipeline.predict(x_test)

    with open('%s/before_after_%s.csv' %
                      (outdir, thesaurus_file), 'a') as outfile:
        outfile.write('DocID,')
        outfile.write(','.join([str(x) for x in range(len(predicted))]))
        outfile.write('\n')
        outfile.write('%s+Thesaurus,' % clf_name)
        outfile.write(','.join([str(x) for x in predicted.tolist()]))
        outfile.write('\n')
        outfile.write('%s-Thesaurus,' % clf_name)
        outfile.write(','.join([str(x) for x in predicted2.tolist()]))
        outfile.write('\n')
        outfile.write('Decisions changed: %d' % (
            nonzero(predicted - predicted2)[0].shape[0]))
        outfile.write('\n')


def _write_exp1_conf_file(base_conf_file, run_id, thes):
    """
    Create a bunch of copies of the base conf file and in the new one modifies
    the experiment name and the (SINGLE) thesaurus used
    """
    log_file, new_conf_file = _prepare_conf_files(base_conf_file, 1,
                                                  run_id)

    replace_in_file(new_conf_file, 'name=.*', 'name=exp%d-%d' % (1, run_id))

    # it is important that the list of thesaurus files in the conf file ends with a comma
    replace_in_file(new_conf_file, 'thesaurus_files=.*',
                    'thesaurus_files=%s,' % thes)
    return new_conf_file, log_file


def _exp1_file_iterator(pattern, base_conf_file):
    """
    Generates a conf file for each thesaurus file matching the provided glob
    pattern. Each conf file is a single classification run
    """
    thesauri = glob.glob(pattern)

    for id, t in enumerate(thesauri):
        new_conf_file, log_file = _write_exp1_conf_file(base_conf_file, id, t)
        yield new_conf_file, log_file
    raise StopIteration


def _prepare_conf_files(base_conf_file, exp_id, run_id):
    """
    Takes a conf files and moves it to a subdirectory to be modified later.
    Also creates a log file for the classification run that will be run with
    the newly created conf file
     Parameter id identifies the clone and distinguishes it from other clones
    """
    name, ext = os.path.splitext(base_conf_file)
    name = '%s-variants' % name
    if not os.path.exists(name):
        os.mkdir(name)
    new_conf_file = os.path.join(name, 'exp%d-%d%s' % (exp_id, run_id, ext))
    log_file = os.path.join(name, '..', 'logs')
    shutil.copy(base_conf_file, new_conf_file)
    configspec_file = os.path.join(os.path.dirname(base_conf_file), '.confrc')
    shutil.copy(configspec_file, name)
    return log_file, new_conf_file


def _write_exp2_to_14_conf_file(base_conf_file, exp_id, run_id, sample_size):
    """
    Prepares conf files for exp2-14 by altering the sample size parameter in
    the
     provided base conf file
    """
    log_dir, new_conf_file = _prepare_conf_files(base_conf_file, exp_id,
                                                 run_id)

    replace_in_file(new_conf_file, 'name=.*',
                    'name=exp%d-%d' % (exp_id, run_id))
    replace_in_file(new_conf_file, 'sample_size=.*',
                    'sample_size=%s' % sample_size)
    return new_conf_file, log_dir


def _exp2_to_14_file_iterator(sizes, exp_id, conf_file):
    for id, size in enumerate(sizes):
        new_conf_file, log_file = _write_exp2_to_14_conf_file(conf_file,
                                                              exp_id,
                                                              id, size)
        print 'Yielding %s, %s' % (new_conf_file, new_conf_file)
        yield new_conf_file, log_file
    raise StopIteration


def _write_exp15_conf_file(base_conf_file, exp_id, run_id, shuffle):
    log_file, new_conf_file = _prepare_conf_files(base_conf_file, exp_id,
                                                  run_id)
    replace_in_file(new_conf_file, 'name=.*',
                    'name=exp%d-%d' % (exp_id, run_id))
    replace_in_file(new_conf_file, 'shuffle_targets=.*',
                    'shuffle_targets=%r' % shuffle)
    return new_conf_file, log_file


def _exp15_file_iterator(conf_file):
    for id, shuffle in enumerate([True, False]):
        new_conf_file, log_file = _write_exp15_conf_file(conf_file, 15, id,
                                                         shuffle)
        print 'Yielding %s, %s' % (new_conf_file, log_file)
        yield new_conf_file, log_file
    raise StopIteration


def _write_exp16_conf_file(base_conf_file, exp_id, run_id, thesauri_list,
                           normalize):
    log_file, new_conf_file = _prepare_conf_files(base_conf_file, exp_id,
                                                  run_id)
    replace_in_file(new_conf_file, 'name=.*',
                    'name=exp%d-%d' % (exp_id, run_id))
    replace_in_file(new_conf_file, 'thesaurus_files=.*',
                    'thesaurus_files=%s' % thesauri_list)
    replace_in_file(new_conf_file, 'normalise_entities=.*',
                    'normalise_entities=%r' % normalize)
    return new_conf_file, log_file


def _exp16_file_iterator(conf_file):
    thesauri = [
        "/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp6-12a/exp6.sims.neighbours.strings,/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp6-12b/exp6.sims.neighbours.strings,/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp6-12c/exp6.sims.neighbours.strings,/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp6-12d/exp6.sims.neighbours.strings,",
        "/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp7-12a/exp7.sims.neighbours.strings,/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp7-12b/exp7.sims.neighbours.strings,/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp7-12c/exp7.sims.neighbours.strings,/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp7-12d/exp7.sims.neighbours.strings,"
    ]
    for id, (thesauri, normalize) in enumerate(
            itertools.product(thesauri, [True, False])):
        new_conf_file, log_file = _write_exp16_conf_file(conf_file, 16, id,
                                                         thesauri, normalize)
        print 'Yielding %s, %s' % (new_conf_file, log_file)
        yield new_conf_file, log_file
    raise StopIteration


def evaluate_thesauri(base_conf_file, file_iterator,
                      reload_data=False, pool_size=1):
    config_obj, configspec_file = parse_config_file(base_conf_file)
    from joblib import Parallel, delayed

    if not reload_data:
        # load the dataset just once
        options = {}
        options['input'] = config_obj['feature_extraction']['input']
        options['shuffle_targets'] = config_obj['shuffle_targets']

        try:
            options['input_generator'] = config_obj['feature_extraction'][
                'input_generator']
        except KeyError:
            options['input_generator'] = ''
        options['source'] = config_obj['training_data']

        print 'Loading training data'
        x_vals, y_vals = _get_data_iterators(**options)
        # todo this only makes sense when we are using a pre-defined test set
        if config_obj['test_data']:
            print 'Loading test data'
            options['source'] = config_obj['test_data']
            x_test, y_test = _get_data_iterators(**options)
        data = (x_vals, y_vals, x_test, y_test)
    else:
        data = None
        print "WARNING: dataset will be reloaded before each sub-experiment"
    Parallel(n_jobs=pool_size)(delayed(go)(new_conf_file, log_file,
                                           data=deepcopy(data)) for
                               new_conf_file, log_file in file_iterator)


def _infer_thesaurus_name(conf_txt):
    thesauri = ''.join(re.findall('thesaurus_files\s*=([\w-]+)', conf_txt))
    if thesauri:
    # thesauri is something like "exp6-11a/exp6.sims.neighbours.strings,"
        corpus = re.findall('exp([0-9]+)', thesauri)[0]
        features = (re.findall('-([0-9]+)', thesauri))[0]
        pos = (re.findall('-[0-9]+(.)', thesauri))[0]
        fef = re.findall('fef([0-9]+)', thesauri)
        # 'fef' isn't in thesaurus name, i.e. has not been postfiltered
        if not fef:
            fef = 0
            print 'WARNING: thesaurus file name %s does not contain ' \
                  'explicit fef information' % thesauri
    else:
        # a thesaurus was not used
        corpus, features, pos, fef = -1, -1, -1, -1

    return corpus, features, fef, pos


def _extract_thesausus_coverage_info(log_txt):
    # token statistics in labelled corpus

    def every_other(iterable):
        """Returns every other element in a iterable in a silly way"""
        return numpy.array(iterable)[range(1, len(iterable), 2)]

    unk_tok = [int(x) for x in every_other(
        re.findall('Unknown tokens: ([0-9]+)', log_txt))]

    total_tok = [int(x) for x in every_other(
        re.findall('Total tokens: ([0-9]+)', log_txt))]
    found_tok = [int(x) for x in every_other(
        re.findall('Found tokens: ([0-9]+)', log_txt))]

    repl_tok = [int(x) for x in every_other(
        re.findall('Replaced tokens: ([0-9]+)', log_txt))]

    total_ty = [int(x) for x in every_other(
        re.findall('Total types: ([0-9]+)', log_txt))]

    unk_ty = [int(x) for x in every_other(
        re.findall('Unknown types: ([0-9]+)', log_txt))]
    found_ty = [int(x) for x in every_other(
        re.findall('Found types: ([0-9]+)', log_txt))]
    repl_ty = [int(x) for x in every_other(
        re.findall('Replaced types: ([0-9]+)', log_txt))]

    # find out how large the thesaurus was from log file
    th_size = re.findall("Thesaurus contains ([0-9]+)", log_txt)
    th_size = int(th_size[0]) if th_size else -1

    return total_tok, total_ty, unk_tok, unk_ty, found_tok, found_ty, \
           repl_tok, repl_ty, th_size


def _pos_statistics(input_file):
    regex1 = re.compile(".*Unknown token.*/(.*)")
    regex2 = re.compile(".*Found thesaurus entry.*/(.*)")
    unknown_pos, found_pos = [], []
    with open(input_file) as infile:
        for line in infile:
            matches = regex1.findall(line)
            if matches:
                unknown_pos.append(matches[0])

            matches = regex2.findall(line)
            if matches:
                found_pos.append(matches[0])

    from collections import Counter

    return Counter(unknown_pos), Counter(found_pos)


def consolidate_results(conf_dir, log_dir, output_dir,
                        unknown_pos_stats_enabled=False):
    """
    Consolidates the results of a series of experiment to ./summary.csv
    A single thesaurus must be used in each experiment
    """
    print 'Consolidating results from %s' % conf_dir
    os.chdir(conf_dir)
    c = csv.writer(open("summary.csv", "w"))
    c.writerow(['name', 'sample_size', 'train_voc_mean', 'train_voc_std',
                'corpus', 'features', 'pos', 'fef',
                'classifier', 'th_size', 'total_tok',
                'unknown_tok_mean', 'unknown_tok_std',
                'found_tok_mean', 'found_tok_std',
                'replaced_tok_mean', 'replaced_tok_std',
                'total_typ',
                'unknown_typ_mean', 'unknown_typ_std',
                'found_typ_mean', 'found_typ_std',
                'replaced_typ_mean', 'replaced_typ_std',
                'metric', 'score_mean', 'score_std'])

    experiments = glob.glob('*.conf')
    unknown_pos_stats, found_pos_stats = {}, {}
    for conf_file in experiments:
        print 'Processing file %s' % conf_file
        with open(conf_file) as infile:
            conf_txt = ''.join(infile.readlines())
        exp_name = re.findall('name=(.*)', conf_txt)[0]

        # find out thesaurus information
        data_shape_x, data_shape_y = [], []
        log_file = os.path.join(log_dir, '%s.log' % exp_name)

        with open(log_file) as infile:
            log_txt = ''.join(infile.readlines())

        lines = re.findall('Total types:', log_txt)
        if not lines:
            print 'WARNING: log file %s does not contain thesaurus ' \
                  'information' % log_file

        sizes = re.findall('Data shape is (\(.*\))', log_txt)
        sizes = [ast.literal_eval(x) for x in sizes]
        # skip the information about the test set
        sizes = numpy.array(sizes)[range(0, len(sizes), 2)]
        for x in sizes:
            data_shape_x.append(x[0])
            data_shape_y.append(x[1])

        if not data_shape_x:
            print "WARNING: training data size not  present in log file %s, " \
                  "trying the other way" \
                  "" % \
                  log_file
            # try the other way of getting the sample size
            try:
                x = re.findall('for each sampling (\d+) documents', log_txt)
                data_shape_x.append(int(x[0]))
            except Exception:
                print "ERROR: that failed too, returning -1"
                data_shape_x.append(-1)


        # find out how many unknown tokens, etc there were from log file
        total_tok, total_ty, unk_tok, unk_ty, found_tok, found_ty, repl_tok, \
        repl_ty, th_size = _extract_thesausus_coverage_info(log_txt)

        # find out the name of the thesaurus(es) from the conf file
        corpus, features, fef, pos = _infer_thesaurus_name(conf_txt)

        def my_mean(x):
            return numpy.mean(x) if x else -1

        def my_std(x):
            return numpy.std(x) if x else -1

        s = int(my_mean(data_shape_x))
        if unknown_pos_stats_enabled:
            unknown_pos_stats[s], found_pos_stats[s] = _pos_statistics(log_file)

        # find out the classifier score from the final csv file
        output_file = os.path.join(output_dir, '%s.out.csv' % exp_name)
        try:
            reader = csv.reader(open(output_file, 'r'))
            _ = reader.next()   # skip over header
            for row in reader:
                classifier, metric, score_my_mean, score_my_std = row

                c.writerow(
                    [exp_name, int(my_mean(data_shape_x)),
                     int(my_mean(data_shape_y)), int(my_std(data_shape_y)),
                     corpus, features, pos, fef, classifier,
                     int(my_mean(th_size)), int(my_mean(total_tok)),
                     int(my_mean(unk_tok)), int(my_std(unk_tok)),
                     int(my_mean(found_tok)), int(my_std(found_tok)),
                     int(my_mean(repl_tok)), int(my_std(repl_tok)),
                     int(my_mean(total_ty)),
                     int(my_mean(unk_ty)), int(my_std(unk_ty)),
                     int(my_mean(found_ty)), int(my_std(found_ty)),
                     int(my_mean(repl_ty)), int(my_std(found_ty)),
                     metric, score_my_mean, score_my_std])
        except IOError:
            print 'WARNING: %s is missing' % output_file
            continue    # file is missing

    if unknown_pos_stats:
        from pandas import DataFrame

        df = DataFrame(unknown_pos_stats).T
        df.to_csv('unknown_token_stats.csv')
        df = DataFrame(found_pos_stats).T
        df.to_csv('found_token_stats.csv')


if __name__ == '__main__':
    i = int(sys.argv[1])
    print 'RUNNING EXPERIMENT %d' % i

    # on local machine
    prefix = '/Volumes/LocalScratchHD/LocalHome/NetBeansProjects/thesisgenerator'
    exp1_thes_pattern = '%s/../Byblo-2.1.0/exp6-1*/*sims.neighbours.strings' % \
                        prefix
    num_workers = 4

    # on cluster
    # prefix = '/mnt/lustre/scratch/inf/mmb28/thesisgenerator'
    # exp1_thes_pattern = '%s/../FeatureExtrationToolkit/exp6-1*/*sims.' \
    #                     'neighbours.strings' % prefix
    # num_workers = 30

    reload_data = False
    # ----------- EXPERIMENT 1 -----------
    if i == 1:
        it = _exp1_file_iterator(exp1_thes_pattern,
                                 '%s/conf/exp1/exp1_base.conf' % prefix)

    # ----------- EXPERIMENTS 2-14 -----------
    elif 1 < i <= 14 or 17 <= i <= 19:
        # sizes = chain(range(100, 1000, 100), range(1000, 5000, 500))
        sizes = chain(range(2, 11, 2), range(20, 101, 10))
        # sizes = range(10, 100, 10)

        base_conf_file = '%s/conf/exp%d/exp%d_base.conf' % (prefix, i, i)
        it = _exp2_to_14_file_iterator(sizes, i, base_conf_file)
    elif i == 15:
        base_conf_file = '%s/conf/exp%d/exp%d_base.conf' % (prefix, i, i)
        it = _exp15_file_iterator(base_conf_file)
        reload_data = True
    elif i == 16:
        base_conf_file = '%s/conf/exp%d/exp%d_base.conf' % (prefix, i, i)
        it = _exp16_file_iterator(base_conf_file)
    else:
        raise ValueError('No such experiment number: %d' % i)

    evaluate_thesauri(base_conf_file, it, pool_size=num_workers,
                      reload_data=reload_data)

    # ----------- CONSOLIDATION -----------
    consolidate_results(
        '%s/conf/exp%d/exp%d_base-variants' % (prefix, i, i),
        '%s/conf/exp%d/logs/' % (prefix, i),
        '%s/conf/exp%d/output/' % (prefix, i)
    )
