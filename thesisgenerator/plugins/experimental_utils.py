# coding=utf-8


# if one tries to run this script from the main project directory the
# thesisgenerator package would not be on the path, add it and try again
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from thesisgenerator.composers.vectorstore import CompositeVectorSource, UnigramVectorSource, \
    PrecomputedSimilaritiesVectorSource
from thesisgenerator.utils.reflection_utils import get_named_object, get_intersection_of_parameters
from thesisgenerator.utils.data_utils import tokenize_data, load_text_data_into_memory, \
    load_tokenizer
from thesisgenerator.utils.conf_file_utils import parse_config_file
from thesisgenerator.utils.misc import get_susx_mysql_conn
from thesisgenerator.plugins.file_generators import _exp16_file_iterator, _exp1_file_iterator, \
    _vary_training_size_file_iterator, get_specific_subexperiment_files

import glob
from itertools import chain
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
                   prefix='/Volumes/LocalDataHD/mmb28/NetBeansProjects/thesisgenerator',
                   vector_source=None):
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
    sizes = chain([2, 5, 10], range(20, 101, 10))
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
    raw_data, data_ids = load_text_data_into_memory(
        training_path=conf['training_data'],
        test_path=conf['test_data'],
        shuffle_targets=conf['shuffle_targets']
    )

    vectors_exist_ = conf['feature_selection']['ensure_vectors_exist']
    handler_ = conf['feature_extraction']['decode_token_handler']
    if 'signified' in handler_.lower() or vectors_exist_:
        # vectors are needed either at decode time (signified handler) or during feature selection
        paths = conf['vector_sources']['unigram_paths']
        precomputed = conf['vector_sources']['precomputed']

        if not paths:
            raise ValueError('You must provide at least one neighbour source because you requested %s '
                             ' and ensure_vectors_exist=%s' % (handler_, vectors_exist_))
        if any('events' in x for x in paths) and precomputed:
            logging.warn('Possible configuration error: you requested precomputed '
                         'thesauri to be used but passed in the following files: \n%s', paths)

        if not precomputed:
            # load unigram vectors and initialise required composers based on these vectors
            if paths:
                logging.info('Loading unigram vector sources')
                unigram_source = UnigramVectorSource(paths,
                                                     reduce_dimensionality=conf['vector_sources'][
                                                         'reduce_dimensionality'],
                                                     dimensions=conf['vector_sources']['dimensions'])

            composers = []
            for section in conf['vector_sources']:
                if 'composer' in section and conf['vector_sources'][section]['run']:
                    # the object must only take keyword arguments
                    composer_class = get_named_object(section)
                    args = get_intersection_of_parameters(composer_class, conf['vector_sources'][section])
                    args['unigram_source'] = unigram_source
                    composers.append(composer_class(**args))
            if composers and not vector_source:
                # if a vector_source has not been predefined
                vector_source = CompositeVectorSource(
                    composers,
                    conf['vector_sources']['sim_threshold'],
                    conf['vector_sources']['include_self'],
                )
        else:
            logging.info('Loading precomputed neighbour sources')
            vector_source = PrecomputedSimilaritiesVectorSource(
                paths,
                conf['vector_sources']['sim_threshold'],
                conf['vector_sources']['include_self'],
            )
    else:
        if not vector_source:
            # if a vector source has not been passed in and has not been initialised, then init it to avoid
            # accessing empty things
            vector_source = []

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
    Parallel(n_jobs=1)(delayed(go)(new_conf_file, log_dir, tokenised_data, vector_source, n_jobs=num_workers) for
                       new_conf_file, log_dir in conf_file_iterator)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s ""(line %(lineno)d)\t%(levelname)s : %(""message)s")

    if len(sys.argv) == 2:
        i = int(sys.argv[1]) # full experiment id
        run_experiment(i)
    elif len(sys.argv) == 3:
        i, j = map(int(sys.argv))
        run_experiment(i, subexpid=j)
    else:
        print 'Expected one or two int parameters, got %s' % (sys.argv)


