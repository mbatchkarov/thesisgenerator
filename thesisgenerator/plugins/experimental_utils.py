# coding=utf-8


# if one tries to run this script from the main project directory the
# thesisgenerator package would not be on the path, add it and try again
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

import glob
import os
import logging
from thesisgenerator.utils.data_utils import get_thesaurus, get_tokenized_data, get_tokenizer_settings_from_conf
from thesisgenerator.utils.conf_file_utils import parse_config_file
from thesisgenerator.plugins.file_generators import _vary_training_size_file_iterator
from thesisgenerator.__main__ import go


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


def run_experiment(expid, num_workers=1,
                   predefined_sized=[],
                   prefix='/mnt/lustre/scratch/inf/mmb28/thesisgenerator',
                   thesaurus=None):
    """

    :param expid: int experiment identified. exp 0 reserved for development purpose and many of the values
      below will be overwriten to ensure the experiment runs (and fails) quickly
    :param num_workers: how many cross-validation runs to do at a time
    :param predefined_sized: how large a sample to take from the data for training
    :param prefix: all output will be written relative to this directory
    :param thesaurus: provide a parsed thesaurus instead of a path to one (useful for unit testing)
    :return:
    """
    logging.info('RUNNING EXPERIMENT %d', expid)

    sizes = [500]  # this is only used if crossval type is subsampled_test_set
    if expid == 0:
        # exp0 is for debugging only, we don't have to do much
        sizes = [180]
        num_workers = 1

    if predefined_sized:
        sizes = predefined_sized

    base_conf_file = '%s/conf/exp%d/exp%d_base.conf' % (prefix, expid, expid)
    conf_file_iterator = _vary_training_size_file_iterator(sizes, expid, base_conf_file)

    _clear_old_files(expid, prefix)
    conf, configspec_file = parse_config_file(base_conf_file)
    if not thesaurus:
        thesaurus = get_thesaurus(conf)

    suffix = '.gz' if conf['joblib_caching'] else ''
    logging.info('Tokenizing %s', conf['training_data'])

    test_path = conf['test_data'] + suffix if conf['test_data'] else ''
    tokenised_data = get_tokenized_data(conf['training_data'] + suffix,
                                        get_tokenizer_settings_from_conf(conf),
                                        test_data=test_path,
                                        gzip_json=conf['joblib_caching'])
    # run data through the pipeline
    return [go(new_conf_file, log_dir, tokenised_data, thesaurus, n_jobs=num_workers)
            for new_conf_file, log_dir in conf_file_iterator]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %(message)s")
    if len(sys.argv) == 2:
        i = int(sys.argv[1])  # full experiment id
        run_experiment(i)
    elif len(sys.argv) == 3:
        i, j = list(map(int(sys.argv)))
        run_experiment(i, subexpid=j)
    else:
        print(('Expected one or two int parameters, got %s' % (sys.argv)))


