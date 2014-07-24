# coding=utf-8


# if one tries to run this script from the main project directory the
# thesisgenerator package would not be on the path, add it and try again
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

import glob
from thesisgenerator.utils.data_utils import (load_text_data_into_memory,
                                              load_tokenizer, get_thesaurus,
                                              tokenize_data)
from thesisgenerator.utils.conf_file_utils import parse_config_file
from thesisgenerator.plugins.file_generators import _vary_training_size_file_iterator
from thesisgenerator.__main__ import go
from thesisgenerator.plugins.dumpers import *


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


def run_experiment(expid, num_workers=4,
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
    print(('RUNNING EXPERIMENT %d' % expid))
    hostname = platform.node()
    if 'apollo' in hostname or 'node' in hostname:
        num_workers = 30

    sizes = [500]
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
    raw_data, data_ids = load_text_data_into_memory(
        training_path=conf['training_data'],
        test_path=conf['test_data'],
        shuffle_targets=conf['shuffle_targets']
    )

    if not thesaurus:
        thesaurus = get_thesaurus(conf)

    tokenizer = load_tokenizer(
        joblib_caching=conf['joblib_caching'],
        normalise_entities=conf['feature_extraction']['normalise_entities'],
        use_pos=conf['feature_extraction']['use_pos'],
        coarse_pos=conf['feature_extraction']['coarse_pos'],
        lemmatize=conf['feature_extraction']['lemmatize'],
        lowercase=conf['tokenizer']['lowercase'],
        remove_stopwords=conf['tokenizer']['remove_stopwords'],
        remove_short_words=conf['tokenizer']['remove_short_words'],
        remove_long_words=conf['tokenizer']['remove_long_words']
    )
    tokenised_data = tokenize_data(raw_data, tokenizer, data_ids)

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


