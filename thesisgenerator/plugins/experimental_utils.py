# coding=utf-8


# if one tries to run this script from the main project directory the
# thesisgenerator package would not be on the path, add it and try again
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from copy import deepcopy
import glob
from itertools import chain
import os
import shutil
import itertools

from numpy import nonzero

from thesisgenerator.__main__ import go, _get_data_iterators, parse_config_file
from thesisgenerator.utils import get_susx_mysql_conn
from thesisgenerator.plugins.dumpers import *
from thesisgenerator.plugins.consolidator import consolidate_results


def _nested_set(dic, key_list, value):
    """
    >>> d = {}
    >>> nested_set(d, ['person', 'address', 'city'], 'New York')
    >>> d
    {'person': {'address': {'city': 'New York'}}}
    """
    for key in key_list[:-1]:
        dic = dic.setdefault(key, {})
    dic[key_list[-1]] = value


def replace_in_file(conf_file, keys, new_value):
    if type(keys) is str:
        # handle the case when there is a single key
        keys = [keys]

    config_obj, configspec_file = parse_config_file(conf_file)
    _nested_set(config_obj, keys, new_value)
    with open(conf_file, 'w') as outfile:
        config_obj.write(outfile)


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

    replace_in_file(new_conf_file, 'name', 'exp%d-%d' % (1, run_id))
    replace_in_file(new_conf_file, 'debug', False)

    # it is important that the list of thesaurus files in the conf file ends with a comma
    replace_in_file(new_conf_file, ['feature_extraction', 'thesaurus_files'],
                    thes)
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
    return log_file, new_conf_file


def _write_exp2_to_14_conf_file(base_conf_file, exp_id, run_id,
                                sample_size):
    """
    Prepares conf files for exp2-14 by altering the sample size parameter in
    the
     provided base conf file
    """
    log_dir, new_conf_file = _prepare_conf_files(base_conf_file, exp_id,
                                                 run_id)

    replace_in_file(new_conf_file, 'name', 'exp%d-%d' % (exp_id, run_id))
    replace_in_file(new_conf_file, ['crossvalidation', 'sample_size'],
                    sample_size)
    # replace_in_file(new_conf_file, 'debug', False)
    return new_conf_file, log_dir


# def _write_exp22conf_file(base_conf_file, exp_id, run_id, sample_size,
#                           keep_only_IT, use_signifier_only):
#     """
#     Prepares conf files for exp22 by altering the sample_size,
#     use_signifier_only and keep_only_IT  parameters in the provided base
#     conf file
#     """
#     log_dir, new_conf_file = _prepare_conf_files(base_conf_file, exp_id,
#                                                  run_id)
#
#     replace_in_file(new_conf_file, 'name', 'exp%d-%d' % (exp_id, run_id))
#     replace_in_file(new_conf_file, ['crossvalidation', 'sample_size'],
#                     sample_size)
#     replace_in_file(new_conf_file, ['feature_extraction', 'use_signifier_only'],
#                     use_signifier_only)
#     replace_in_file(new_conf_file, ['tokenizer', 'keep_only_IT'], keep_only_IT)
#     replace_in_file(new_conf_file, 'debug', False)
#     return new_conf_file, log_dir


def _vary_training_size_file_iterator(sizes, exp_id, conf_file):
    for sub_id, size in enumerate(sizes):
        new_conf_file, log_file = _write_exp2_to_14_conf_file(conf_file,
                                                              exp_id,
                                                              sub_id,
                                                              size)
        print 'Yielding %s, %s' % (new_conf_file, new_conf_file)
        yield new_conf_file, log_file
    raise StopIteration


# def _exp22_file_iterator(sizes, exp_id, conf_file):
#     sub_id = 0
#     for size in sizes:
#         for keep_only_IT in [True, False]:
#             for use_signifier_only in [True, False]:
#                 new_conf_file, log_file = _write_exp22conf_file(conf_file,
#                                                                 exp_id,
#                                                                 sub_id, size,
#                                                                 keep_only_IT,
#                                                                 use_signifier_only)
#                 print 'Yielding %s, %s' % (new_conf_file, new_conf_file)
#                 yield new_conf_file, log_file
#                 sub_id += 1
#     raise StopIteration


def _write_exp15_conf_file(base_conf_file, exp_id, run_id, shuffle):
    log_file, new_conf_file = _prepare_conf_files(base_conf_file, exp_id,
                                                  run_id)
    replace_in_file(new_conf_file, 'name',
                    'exp%d-%d' % (exp_id, run_id))
    replace_in_file(new_conf_file, 'shuffle_targets',
                    'shuffle_targets' % shuffle)
    replace_in_file(new_conf_file, 'debug', False)
    return new_conf_file, log_file


def _exp15_file_iterator(conf_file):
    for id, shuffle in enumerate([True, False]):
        new_conf_file, log_file = _write_exp15_conf_file(conf_file, 15,
                                                         id, shuffle)
        print 'Yielding %s, %s' % (new_conf_file, log_file)
        yield new_conf_file, log_file
    raise StopIteration


def _write_exp16_conf_file(base_conf_file, exp_id, run_id,
                           thesauri_list,
                           normalize):
    log_file, new_conf_file = _prepare_conf_files(base_conf_file, exp_id,
                                                  run_id)
    replace_in_file(new_conf_file, 'name',
                    'exp%d-%d' % (exp_id, run_id))
    replace_in_file(new_conf_file, ['feature_extraction', 'thesaurus_files'],
                    thesauri_list)
    replace_in_file(new_conf_file, ['feature_extraction', 'normalise_entities'],
                    normalize)
    replace_in_file(new_conf_file, 'debug', False)
    return new_conf_file, log_file


def _exp16_file_iterator(conf_file):
    thesauri = [
        "/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp6-12a/exp6.sims.neighbours.strings,/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp6-12b/exp6.sims.neighbours.strings,/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp6-12c/exp6.sims.neighbours.strings,/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp6-12d/exp6.sims.neighbours.strings,",
        "/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp7-12a/exp7.sims.neighbours.strings,/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp7-12b/exp7.sims.neighbours.strings,/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp7-12c/exp7.sims.neighbours.strings,/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp7-12d/exp7.sims.neighbours.strings,"
    ]
    for id, (thesauri, normalize) in enumerate(
            itertools.product(thesauri, [True, False])):
        new_conf_file, log_file = _write_exp16_conf_file(conf_file,
                                                         16, id,
                                                         thesauri, normalize)
        print 'Yielding %s, %s' % (new_conf_file, log_file)
        yield new_conf_file, log_file
    raise StopIteration


def evaluate_thesauri(base_conf_file, file_iterator,
                      reload_data=False, pool_size=1):
    config_obj, configspec_file = parse_config_file(base_conf_file)
    from joblib import Parallel, delayed

    if reload_data:
        data = None
        print "WARNING: raw dataset will be reloaded before each " \
              "sub-experiment"
    else:
        # read the raw text just once
        options = {'input': config_obj['feature_extraction']['input'],
                   'shuffle_targets': config_obj['shuffle_targets']}

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

    Parallel(n_jobs=pool_size)(delayed(go)(new_conf_file, log_file,
                                           data=deepcopy(data)) for
                               new_conf_file, log_file in file_iterator)


def _clear_old_files(i, prefix):
    """
    clear old conf, logs and result files for this experiment
    """
    for f in glob.glob('%s/conf/exp%d/exp%d_base-variants/*' % (prefix, i, i)):
        os.remove(f)
    for f in glob.glob('%s/conf/exp%d/output/*' % (prefix, i)):
        os.remove(f)
    for f in glob.glob('%s/conf/exp%d/logs/*' % (prefix, i)):
        os.remove(f)


def run_experiment(i, num_workers=4,
                   predefined_sized=[],
                   prefix='/Volumes/LocalScratchHD/LocalHome/NetBeansProjects/thesisgenerator'):
    print 'RUNNING EXPERIMENT %d' % i
    # on local machine

    exp1_thes_pattern = '%s/../Byblo-2.1.0/exp6-1*/*sims.neighbours.strings' % \
                        prefix
    # on cluster
    import platform

    hostname = platform.node()
    if 'apollo' in hostname or 'node' in hostname:
        # set the paths automatically when on the cluster
        prefix = '/mnt/lustre/scratch/inf/mmb28/thesisgenerator'
        exp1_thes_pattern = '%s/../FeatureExtrationToolkit/exp6-1*/*sims.' \
                            'neighbours.strings' % prefix
        num_workers = 30

    sizes = chain(range(2, 11, 2), range(20, 101, 10))
    if i == 0:
        # exp0 is for debugging only, we don't have to do much
        sizes = range(10, 31, 10)
    if predefined_sized:
        sizes = predefined_sized

    reload_data = False
    # ----------- EXPERIMENT 1 -----------
    base_conf_file = '%s/conf/exp%d/exp%d_base.conf' % (prefix, i, i)
    if i == 1:
        it = _exp1_file_iterator(exp1_thes_pattern,
                                 '%s/conf/exp1/exp1_base.conf' % prefix)

    # ----------- EXPERIMENTS 2-14 -----------
    elif i == 0 or 1 < i <= 14 or 17 <= i <= 25:
        it = _vary_training_size_file_iterator(sizes, i, base_conf_file)
    elif i == 15:
        it = _exp15_file_iterator(base_conf_file)
        reload_data = True
    elif i == 16:
        it = _exp16_file_iterator(base_conf_file)
    else:
        raise ValueError('No such experiment number: %d' % i)

    _clear_old_files(i, prefix)
    evaluate_thesauri(base_conf_file, it, pool_size=num_workers,
                      reload_data=reload_data)

    # ----------- CONSOLIDATION -----------
    output_dir = '%s/conf/exp%d/output/' % (prefix, i)
    csv_out_fh = open(os.path.join(output_dir, "summary%d.csv" % i), "w")

    if not ('apollo' in hostname or 'node' in hostname):
        output_db_conn = get_susx_mysql_conn()
        writer = ConsolidatedResultsSqlAndCsvWriter(i, csv_out_fh,
                                                    output_db_conn)
    else:
        writer = ConsolidatedResultsCsvWriter(csv_out_fh)
    consolidate_results(
        writer,
        '%s/conf/exp%d/exp%d_base-variants' % (prefix, i, i),
        '%s/conf/exp%d/logs/' % (prefix, i),
        output_dir
    )


if __name__ == '__main__':
    run_experiment(int(sys.argv[1]))
