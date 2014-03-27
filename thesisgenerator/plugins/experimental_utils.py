# coding=utf-8


# if one tries to run this script from the main project directory the
# thesisgenerator package would not be on the path, add it and try again
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

import platform
import glob
from itertools import chain
from thesisgenerator.utils.data_utils import tokenize_data, load_text_data_into_memory, \
    load_tokenizer, get_vector_source
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


def run_experiment(expid, subexpid=None, num_workers=4,
                   predefined_sized=[],
                   prefix='/mnt/lustre/scratch/inf/mmb28/thesisgenerator',
                   vector_source=None):
    print('RUNNING EXPERIMENT %d' % expid)
    hostname = platform.node()
    if 'apollo' in hostname or 'node' in hostname:
        num_workers = 30

    sizes = chain(range(10, 101, 15), range(200, 501, 100))
    if expid == 0:
        # exp0 is for debugging only, we don't have to do much
        sizes = [5]  #range(10, 31, 10)
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

    vector_source = get_vector_source(conf, vector_source=vector_source)

    tokenizer = load_tokenizer(
        joblib_caching=conf['joblib_caching'],
        normalise_entities=conf['feature_extraction']['normalise_entities'],
        use_pos=conf['feature_extraction']['use_pos'],
        coarse_pos=conf['feature_extraction']['coarse_pos'],
        lemmatize=conf['feature_extraction']['lemmatize'],
        lowercase=conf['tokenizer']['lowercase'],
        remove_stopwords=conf['tokenizer']['remove_stopwords'],
        remove_short_words=conf['tokenizer']['remove_short_words'])
    tokenised_data = tokenize_data(raw_data, tokenizer, data_ids)

    # run data through the pipeline
    return [go(new_conf_file, log_dir, tokenised_data, vector_source, n_jobs=num_workers)
            for new_conf_file, log_dir in conf_file_iterator]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %(message)s")

    if len(sys.argv) == 2:
        i = int(sys.argv[1])  # full experiment id
        run_experiment(i)
    elif len(sys.argv) == 3:
        i, j = map(int(sys.argv))
        run_experiment(i, subexpid=j)
    else:
        print 'Expected one or two int parameters, got %s' % (sys.argv)


