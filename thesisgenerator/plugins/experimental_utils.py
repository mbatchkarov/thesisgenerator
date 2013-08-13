# coding=utf-8


# if one tries to run this script from the main project directory the
# thesisgenerator package would not be on the path, add it and try again
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from thesisgenerator.utils.data_utils import tokenize_data, load_text_data_into_memory, _init_utilities_state
from thesisgenerator.utils.conf_file_utils import parse_config_file
from thesisgenerator.utils.misc import get_susx_mysql_conn
from thesisgenerator.plugins.file_generators import _exp16_file_iterator, _exp1_file_iterator, \
    _vary_training_size_file_iterator, get_specific_subexperiment_files

import glob
from itertools import chain
import os
from joblib import Parallel, delayed

from numpy import nonzero

from thesisgenerator.__main__ import go
from thesisgenerator.plugins.dumpers import *
from thesisgenerator.plugins.consolidator import consolidate_results


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
                   prefix='/Volumes/LocalDataHD/mmb28/NetBeansProjects/thesisgenerator'):
    print 'RUNNING EXPERIMENT %d' % expid
    # on local machine

    exp1_thes_pattern = '%s/../Byblo-2.1.0/exp6-1*/*sims.neighbours.strings' % prefix
    # on cluster
    import platform

    hostname = platform.node()
    if 'apollo' in hostname or 'node' in hostname:
        # set the paths automatically when on the cluster
        prefix = '/mnt/lustre/scratch/inf/mmb28/thesisgenerator'
        exp1_thes_pattern = '%s/../FeatureExtrationToolkit/exp6-1*/*sims.neighbours.strings' % prefix
        num_workers = 30

    if subexpid:
        # requested a sub-experiment
        new_conf_file, log_file = get_specific_subexperiment_files(expid, subexpid)
        return

    # requested a whole experiment, with a bunch of training data set sizes
    sizes = chain(range(2, 11, 2), range(20, 101, 10))
    if expid == 0:
        # exp0 is for debugging only, we don't have to do much
        sizes = [10, 20]#range(10, 31, 10)
        num_workers = 1
    if predefined_sized:
        sizes = predefined_sized

    # ----------- EXPERIMENT 1 -----------
    base_conf_file = '%s/conf/exp%d/exp%d_base.conf' % (prefix, expid, expid)
    if expid == 1:
        conf_file_iterator = _exp1_file_iterator(exp1_thes_pattern, '%s/conf/exp1/exp1_base.conf' % prefix)

    # ----------- EXPERIMENTS 2-14 -----------
    elif expid == 0 or 1 < expid <= 14 or 17 <= expid:
        conf_file_iterator = _vary_training_size_file_iterator(sizes, expid, base_conf_file)
    elif expid == 16:
        conf_file_iterator = _exp16_file_iterator(base_conf_file)
    else:
        raise ValueError('No such experiment number: %d' % expid)

    _clear_old_files(expid, prefix)
    conf, configspec_file = parse_config_file(base_conf_file)
    raw_data = load_text_data_into_memory(conf)
    thesaurus, tokenizer = _init_utilities_state(conf)
    keep_only_IT = conf['tokenizer']['keep_only_IT']
    tokenised_data = tokenize_data(raw_data, tokenizer, keep_only_IT)

    # run the data through the pipeline
    Parallel(n_jobs=num_workers)(delayed(go)(new_conf_file, log_dir, tokenised_data, thesaurus) for
                                 new_conf_file, log_dir in conf_file_iterator)

    # ----------- CONSOLIDATION -----------
    output_dir = '%s/conf/exp%d/output/' % (prefix, expid)
    csv_out_fh = open(os.path.join(output_dir, "summary%d.csv" % expid), "w")

    if not ('apollo' in hostname or 'node' in hostname):
        output_db_conn = get_susx_mysql_conn()
        writer = ConsolidatedResultsSqlAndCsvWriter(expid, csv_out_fh, output_db_conn)
    else:
        writer = ConsolidatedResultsCsvWriter(csv_out_fh)
    consolidate_results(
        writer,
        '%s/conf/exp%d/exp%d_base-variants' % (prefix, expid, expid),
        '%s/conf/exp%d/logs/' % (prefix, expid),
        output_dir
    )


if __name__ == '__main__':
    if len(sys.argv) == 2:
        i = int(sys.argv[1]) # full experiment id
        run_experiment(i)
    elif len(sys.argv) == 3:
        i, j = map(int(sys.argv))
        run_experiment(i, subexpid=j)
    else:
        print 'Expected one or two int parameters, got %s' % (sys.argv)


