import os
import shutil
from discoutils.misc import mkdirs_if_not_exists
from thesisgenerator.utils.conf_file_utils import set_in_conf_file, parse_config_file


def vary_training_size_file_iterator(sizes, exp_id, base_conf_file):
    for sub_id, size in enumerate(sizes):
        new_conf_file, log_file = _copy_conf_file(base_conf_file,
                                                              exp_id,
                                                              sub_id,
                                                              size)
        yield new_conf_file, log_file
    raise StopIteration


def _copy_conf_file(base_conf_file, exp_id, run_id,
                                sample_size):
    """
    Prepares conf files for exp2-14 by altering the sample size parameter in
    the provided base conf file
    """
    log_dir, new_conf_file = _prepare_conf_files(base_conf_file, exp_id, run_id)

    config_obj, _ = parse_config_file(base_conf_file)
    old_name = config_obj['name']

    set_in_conf_file(new_conf_file, 'name', '%s-%d' % (old_name, run_id))
    set_in_conf_file(new_conf_file, ['crossvalidation', 'sample_size'],
                     sample_size)
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
    mkdirs_if_not_exists(name)
    new_conf_file = os.path.join(name, 'exp%d-%d%s' % (exp_id, run_id, ext))
    log_file = os.path.join(name, '..', 'logs')
    shutil.copy(base_conf_file, new_conf_file)
    return log_file, new_conf_file