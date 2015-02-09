import os
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
from dissect_scripts.load_translated_byblo_space import train_baroni_composer
from glob import glob
import argparse
import logging
from discoutils.misc import mkdirs_if_not_exists
from discoutils.cmd_utils import (set_stage_in_byblo_conf_file, run_byblo, parse_byblo_conf_file,
                                  unindex_all_byblo_vectors)
from discoutils.reweighting import ppmi_sparse_matrix
from discoutils.reduce_dimensionality import do_svd
from discoutils.misc import temp_chdir
from discoutils.thesaurus_loader import Vectors
from thesisgenerator.composers.vectorstore import (AdditiveComposer, MultiplicativeComposer,
                                                   LeftmostWordComposer, RightmostWordComposer,
                                                   BaroniComposer, compose_and_write_vectors)

"""
Composed wiki/gigaw dependency/window vectors and writes them to FeatureExtractionToolkit/exp10-13-composed-ngrams
"""
prefix = '/mnt/lustre/scratch/inf/mmb28/FeatureExtractionToolkit/'
byblo_base_dir = os.path.join(prefix, 'Byblo-2.2.0')


def calculate_unigram_vectors(thesaurus_dir):
    # find conf file in directory
    byblo_conf_file = _find_conf_file(thesaurus_dir)

    # find out where the conf file said output should go
    opts, _ = parse_byblo_conf_file(byblo_conf_file)
    byblo_output_prefix = os.path.join(opts.output, os.path.basename(opts.input))

    # get byblo to calculate vectors for all entries
    set_stage_in_byblo_conf_file(byblo_conf_file, 1)
    run_byblo(byblo_conf_file)
    set_stage_in_byblo_conf_file(byblo_conf_file, 0)
    # get vectors as strings
    unindex_all_byblo_vectors(byblo_output_prefix)


def _find_conf_file(thesaurus_dir):
    return glob(os.path.join(thesaurus_dir, '*conf*'))[0]


def _find_allpairs_file(thesaurus_dir):
    return [x for x in glob(os.path.join(thesaurus_dir, '*sims.neighbours.strings')) if 'svd' not in x.lower()][0]


def _find_events_file(thesaurus_dir):
    return [x for x in glob(os.path.join(thesaurus_dir, '*events.filtered.strings')) if 'svd' not in x.lower()][0]


def _find_output_prefix(thesaurus_dir):
    # todo this will not work if we're throwing multiple events/features/entries files (eg SVD reduced and non-reduced)
    # into the same directory
    return os.path.commonprefix(
        [x for x in glob(os.path.join(thesaurus_dir, '*filtered*')) if 'svd' not in x.lower()])[:-1]


def _do_ppmi(vectors_path, output_dir):
    v = Vectors.from_tsv(vectors_path)
    ppmi_sparse_matrix(v.matrix)
    v.to_tsv(os.path.join(output_dir, os.path.basename(vectors_path)), gzipped=True)


def build_unreduced_counting_thesauri(corpus, corpus_name, features,
                                      stages, use_ppmi):
    """
    required files: a Byblo conf file, a labelled classification data set
    created files:  composed vector files in a dir, thesauri of NPs
    """

    # SET UP A FEW REQUIRED PATHS
    unigram_thesaurus_dir = os.path.abspath(os.path.join(prefix,
                                                         'exp%d-%db' % (corpus, features)))

    unigram_thesaurus_dir_ppmi = os.path.abspath(os.path.join(prefix,
                                                              'exp%d-%db-ppmi' % (corpus, features)))
    mkdirs_if_not_exists(unigram_thesaurus_dir_ppmi)

    ngram_vectors_dir = os.path.join(prefix,
                                     'exp%d-%d-composed-ngrams' % (corpus, features))
    ngram_vectors_dir_ppmi = os.path.join(prefix,
                                          'exp%d-%d-composed-ngrams-ppmi' % (corpus, features))

    composer_algos = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer,
                      RightmostWordComposer]  # observed done through a separate script

    mkdirs_if_not_exists(ngram_vectors_dir)
    mkdirs_if_not_exists(ngram_vectors_dir_ppmi)

    # EXTRACT UNIGRAM VECTORS WITH BYBLO
    if 'unigrams' in stages:
        with temp_chdir(byblo_base_dir):
            calculate_unigram_vectors(unigram_thesaurus_dir)
    else:
        logging.warning('Skipping unigrams stage. Assuming output is at %s',
                        _find_events_file(unigram_thesaurus_dir))

    if 'ppmi' in stages:
        _do_ppmi(_find_events_file(unigram_thesaurus_dir),
                 unigram_thesaurus_dir_ppmi)

    # COMPOSE ALL AN/NN VECTORS IN LABELLED SET
    if 'compose' in stages:
        if use_ppmi:
            unigram_vectors_file = _find_events_file(unigram_thesaurus_dir_ppmi)
            outdir = ngram_vectors_dir_ppmi
        else:
            unigram_vectors_file = _find_events_file(unigram_thesaurus_dir)
            outdir = ngram_vectors_dir
        compose_and_write_vectors(unigram_vectors_file,
                                  corpus_name,
                                  composer_algos,
                                  output_dir=outdir)
    else:
        logging.warning('Skipping composition stage. Assuming output is at %s', ngram_vectors_dir)


def build_full_composed_thesauri_with_baroni_and_svd(corpus, features, stages):
    # SET UP A FEW REQUIRED PATHS
    global prefix, byblo_base_dir
    # INPUT 1:  DIRECTORY. Must contain a single conf file
    unigram_thesaurus_dir = os.path.join(prefix, 'exp%d-%db' % (corpus, features))

    # INPUT 2: A FILE, TSV, underscore-separated observed vectors for ANs and NNs
    target_dimensionality = [100]
    dataset_name = 'gigaw' if corpus == 10 else 'wiki'  # short name of input corpus
    features_name = 'wins' if features == 13 else 'deps'  # short name of input corpus
    baroni_training_phrase_types = {'AN', 'NN'}  # what kind of NPs to train Baroni composer for
    baroni_training_phrases = os.path.join(prefix, 'observed_vectors',
                                           '%s_NPs_%s_observed' % (dataset_name, features_name))
    ngram_vectors_dir = os.path.join(prefix,
                                     'exp%d-%d-composed-ngrams' % (corpus, features))  # output 1
    if features_name == 'wins':
        composer_algos = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer,
                          RightmostWordComposer, BaroniComposer]
    else:
        # can't train Baroni on deps because I don't have observed vectors for phrases
        composer_algos = [AdditiveComposer, MultiplicativeComposer,
                          LeftmostWordComposer, RightmostWordComposer]

    mkdirs_if_not_exists(ngram_vectors_dir)

    # EXTRACT UNIGRAM VECTORS WITH BYBLO
    if 'unigrams' in stages:
        with temp_chdir(byblo_base_dir):
            calculate_unigram_vectors(unigram_thesaurus_dir)
    else:
        logging.warning('Skipping unigrams stage. Assuming output is at %s',
                        _find_events_file(unigram_thesaurus_dir))

    # REDUCE DIMENSIONALITY
    # add in observed AN/NN vectors for SVD processing. Reduce both unigram vectors and observed phrase vectors
    # together and put the output into the same file
    unreduced_unigram_events_file = _find_events_file(unigram_thesaurus_dir)
    # ...exp6-12/exp6.events.filtered.strings --> ...exp6-12/exp6
    reduced_file_prefix = os.path.join(unigram_thesaurus_dir,
                                       'exp%d-with-obs-phrases' % corpus)
    # only keep the most frequent types per PoS tag to speed things up
    counts = [('N', 20000), ('V', 0), ('J', 10000), ('RB', 0), ('AN', 0), ('NN', 0)]
    if 'svd' in stages:
        if features_name == 'deps':
            # havent got observed vectors for these, do SVD on the unigrams only
            do_svd(unreduced_unigram_events_file, reduced_file_prefix,
                   desired_counts_per_feature_type=counts, reduce_to=target_dimensionality,
                   write=1)
        else:
            # in this case the name exp%d-with-obs-phrases is massively misleading because
            # there aren't any obs phrase vectors
            # let's just do SVD on the unigram phrases so we can compose them simply later
            do_svd(unreduced_unigram_events_file, reduced_file_prefix,
                   desired_counts_per_feature_type=counts, reduce_to=target_dimensionality,
                   apply_to=baroni_training_phrases)
    else:
        logging.warning('Skipping SVD stage. Assuming output is at %s-SVD*', reduced_file_prefix)

    # construct the names of files output by do_svd
    baroni_training_data = ['%s-SVD%d.events.filtered.strings' % (reduced_file_prefix, dim)
                            for dim in target_dimensionality]

    trained_composer_files = []
    # TRAIN BARONI COMPOSER
    # train on each SVD-reduced file, not the original one, one composer object for both AN and NN phrases
    for svd_dims, all_reduced_vectors in zip(target_dimensionality, baroni_training_data):
        # first set up paths
        baroni_training_heads = '%s-onlyN-SVD%s.tmp' % (baroni_training_phrases, svd_dims)
        baroni_training_only_phrases = '%s-onlyPhrases-SVD%s.tmp' % (baroni_training_phrases, svd_dims)
        trained_composer_prefix = '%s-SVD%s' % (baroni_training_phrases, svd_dims)
        trained_composer_file = trained_composer_prefix + '.composer.pkl'
        trained_composer_files.append(trained_composer_file)

        if 'baroni' in stages and features_name != 'deps':
            # do the actual training
            thes = Vectors.from_tsv(all_reduced_vectors, lowercasing=False)
            thes.to_tsv(baroni_training_heads,
                        entry_filter=lambda x: x.type == '1-GRAM' and x.tokens[0].pos == 'N')

            thes.to_tsv(baroni_training_only_phrases,
                        entry_filter=lambda x: x.type in baroni_training_phrase_types,
                        row_transform=lambda x: x.replace(' ', '_'))

            train_baroni_composer(baroni_training_heads,
                                  baroni_training_only_phrases,
                                  trained_composer_prefix,
                                  threshold=50)


        else:
            logging.warning('Skipping Baroni training stage. Assuming trained models are at %s', trained_composer_files)

    if 'compose' in stages:
        for svd_dims, all_reduced_vectors, trained_composer_file \
                in zip(target_dimensionality, baroni_training_data, trained_composer_files):
            # it is OK for the first parameter to contain phrase vectors, there is explicit filtering coming up
            # the assumption is these are actually observed phrasal vectors
            compose_and_write_vectors(all_reduced_vectors,
                                      '%s-%s' % (dataset_name, svd_dims),
                                      composer_algos,
                                      pretrained_Baroni_composer_file=trained_composer_file,
                                      output_dir=ngram_vectors_dir)
    else:
        logging.warning('Skipping composition stage. Assuming output is at %s', ngram_vectors_dir)


def get_corpus_features_cmd_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--corpus', choices=('wikipedia', 'gigaword'), required=True,
                        help='Unlabelled corpus to source unigram vectors from')
    parser.add_argument('--features', choices=('dependencies', 'windows'), required=True,
                        help='Feature type of unigram vectors')
    return parser


def get_cmd_parser():
    parser = argparse.ArgumentParser(parents=[get_corpus_features_cmd_parser()])
    # add options specific to this script here
    parser.add_argument('--stages', choices=('unigrams', 'ppmi', 'svd', 'baroni', 'compose', 'symlink'),
                        required=True,
                        nargs='+',
                        help='What parts of the pipeline to run. Each part is independent, the pipeline can be '
                             'run incrementally. The stages are: '
                             '1) unigrams: extract unigram vectors from unlabelled corpus '
                             '2) ppmi: perform PPMI reweighting on feature counts '
                             '3) svd: reduce noun and adj matrix, apply to NP matrix '
                             '4) baroni: train Baroni composer '
                             '5) compose: compose all possible NP vectors with all composers ')
    parser.add_argument('--use-svd', action='store_true',
                        help='If set, SVD will be performed and a Baroni composer will be trained. Otherwise the'
                             'svd part of the pipeline is skipped.')
    parser.add_argument('--use-ppmi', action='store_true',
                        help='If set, PPMI will be performed. Currently this is only implemented without SVD')
    return parser


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %(message)s")
    parameters = get_cmd_parser().parse_args()
    logging.info(parameters)

    corpus_name = parameters.corpus[:5]  # human-readable corpus name
    corpus = {'gigaword': 10, 'wikipedia': 11}
    features = {'dependencies': 12, 'windows': 13}
    if parameters.use_svd:
        logging.info('Starting pipeline with SVD and Baroni composer')
        build_full_composed_thesauri_with_baroni_and_svd(corpus[parameters.corpus], features[parameters.features],
                                                         parameters.stages)
    else:
        logging.info('Starting pipeline without SVD')
        build_unreduced_counting_thesauri(corpus[parameters.corpus], corpus_name, features[parameters.features],
                                          parameters.stages, parameters.use_ppmi)

