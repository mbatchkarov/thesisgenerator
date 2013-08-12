import glob
import itertools
import os
import shutil
from thesisgenerator.utils.conf_file_utils import set_in_conf_file

__author__ = 'mmb28'


def _exp15_file_iterator(conf_file):
    for id, shuffle in enumerate([True, False]):
        new_conf_file, log_file = _write_exp15_conf_file(conf_file, 15,
                                                         id, shuffle)
        print 'Yielding %s, %s' % (new_conf_file, log_file)
        yield new_conf_file, log_file
    raise StopIteration


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


def _vary_training_size_file_iterator(sizes, exp_id, conf_file):
    for sub_id, size in enumerate(sizes):
        new_conf_file, log_file = _write_exp2_to_14_conf_file(conf_file,
                                                              exp_id,
                                                              sub_id,
                                                              size)
        print 'Yielding %s, %s' % (new_conf_file, new_conf_file)
        yield new_conf_file, log_file
    raise StopIteration


def get_specific_subexperiment_files(id, subid):
    pass


def _write_exp15_conf_file(base_conf_file, exp_id, run_id, shuffle):
    log_file, new_conf_file = _prepare_conf_files(base_conf_file, exp_id,
                                                  run_id)
    set_in_conf_file(new_conf_file, 'name',
                     'exp%d-%d' % (exp_id, run_id))
    set_in_conf_file(new_conf_file, 'shuffle_targets',
                     'shuffle_targets' % shuffle)
    set_in_conf_file(new_conf_file, 'debug', False)
    return new_conf_file, log_file


def _write_exp16_conf_file(base_conf_file, exp_id, run_id,
                           thesauri_list,
                           normalize):
    log_file, new_conf_file = _prepare_conf_files(base_conf_file, exp_id,
                                                  run_id)
    set_in_conf_file(new_conf_file, 'name',
                     'exp%d-%d' % (exp_id, run_id))
    set_in_conf_file(new_conf_file, ['feature_extraction', 'thesaurus_files'],
                     thesauri_list)
    set_in_conf_file(new_conf_file, ['feature_extraction', 'normalise_entities'],
                     normalize)
    set_in_conf_file(new_conf_file, 'debug', False)
    return new_conf_file, log_file


def _write_exp1_conf_file(base_conf_file, run_id, thes):
    """
    Create a bunch of copies of the base conf file and in the new one modifies
    the experiment name and the (SINGLE) thesaurus used
    """
    log_file, new_conf_file = _prepare_conf_files(base_conf_file, 1,
                                                  run_id)

    set_in_conf_file(new_conf_file, 'name', 'exp%d-%d' % (1, run_id))
    set_in_conf_file(new_conf_file, 'debug', False)

    # it is important that the list of thesaurus files in the conf file ends with a comma
    set_in_conf_file(new_conf_file, ['feature_extraction', 'thesaurus_files'],
                     thes)
    return new_conf_file, log_file


def _write_exp2_to_14_conf_file(base_conf_file, exp_id, run_id,
                                sample_size):
    """
    Prepares conf files for exp2-14 by altering the sample size parameter in
    the
     provided base conf file
    """
    log_dir, new_conf_file = _prepare_conf_files(base_conf_file, exp_id,
                                                 run_id)

    set_in_conf_file(new_conf_file, 'name', 'exp%d-%d' % (exp_id, run_id))
    set_in_conf_file(new_conf_file, ['crossvalidation', 'sample_size'],
                     sample_size)
    # replace_in_file(new_conf_file, 'debug', False)
    return new_conf_file, log_dir


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