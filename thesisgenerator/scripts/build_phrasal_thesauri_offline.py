import logging
import os
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
from dissect_scripts.load_translated_byblo_space import train_baroni_composer
from glob import glob
import shutil
import argparse
from discoutils.io_utils import write_vectors_to_disk
from discoutils.misc import Bunch
from discoutils.thesaurus_loader import Thesaurus
from discoutils.cmd_utils import set_stage_in_byblo_conf_file, run_byblo, parse_byblo_conf_file, \
    reindex_all_byblo_vectors, run_and_log_output, unindex_all_byblo_vectors, set_output_in_byblo_conf_file
from discoutils.reduce_dimensionality import do_svd
from thesisgenerator.scripts import dump_all_composed_vectors as dump
from thesisgenerator.composers.vectorstore import *


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


def add_new_vectors(thesaurus_dir, composed_vectors, composed_entries, composed_features):
    # todo this appending should only occur once, should I check for it?

    entries = glob(os.path.join(thesaurus_dir, '*.entries.filtered.strings'))[0]
    run_and_log_output('cat {} >> {}'.format(composed_entries, entries))
    events = glob(os.path.join(thesaurus_dir, '*.events.filtered.strings'))[0]
    run_and_log_output('cat {} >> {}'.format(composed_vectors, events))

    # features of newly created entries must be the same as these of the old ones, but let's check just in case
    # Byblo throws an out-of-bounds exception if features do not match
    features_file = glob(os.path.join(thesaurus_dir, '*.features.filtered.strings'))[0]
    with open('{}'.format(features_file)) as infile:
        old_features = set(x.strip().split('\t')[0] for x in infile.readlines())
    with open(composed_features) as infile:
        new_features = [x.strip().split('\t') for x in infile.readlines()]

    with open(features_file, 'a+b') as outfile:  # append mode
        for feature, count in new_features:
            if feature not in old_features:
                outfile.write('%s\t%s\n' % (feature, count))
            else:
                logging.debug('Ignoring duplicate feature %s with count %s', feature, count)


def _find_new_files(feature_type, directory):
    res = \
        glob(os.path.join(directory, '%s*vectors.tsv' % feature_type))[0], \
        glob(os.path.join(directory, '%s*entries.txt' % feature_type))[0], \
        glob(os.path.join(directory, '%s*features.txt' % feature_type))[0]
    return res


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


def do_second_part(thesaurus_dir, vectors_files='', entries_files='', features_files='', copy_to_dir=None):
    if copy_to_dir:
        assert vectors_files
        assert entries_files
        assert features_files

        if os.path.exists(copy_to_dir):
            shutil.rmtree(copy_to_dir)  # copytree will fail if target exists
        shutil.copytree(thesaurus_dir, copy_to_dir)

        add_new_vectors(copy_to_dir, vectors_files, entries_files, features_files)
        thesaurus_dir = copy_to_dir

    # restore indices from strings
    thes_prefix = _find_output_prefix(thesaurus_dir)
    reindex_all_byblo_vectors(thes_prefix)

    # re-run all-pairs similarity
    # first change output prefix in conf file, in case the if-statement above has made a new working directory
    byblo_conf_file = _find_conf_file(thesaurus_dir)
    # tell byblo to only do the later stages
    set_output_in_byblo_conf_file(byblo_conf_file, thesaurus_dir)
    set_stage_in_byblo_conf_file(byblo_conf_file, 2)
    run_byblo(byblo_conf_file)
    set_stage_in_byblo_conf_file(byblo_conf_file, 0)


def do_second_part_without_base_thesaurus(byblo_conf_file, output_dir, vectors_file='', entries_file='',
                                          features_file=''):
    '''
    Takes a set of plain-text TSV files and builds a thesaurus out of them
    :param byblo_conf_file: File that specifies filtering, similarity measure, etc
    :param output_dir: where should the output go
    :param vectors_file:
    :param entries_file:
    :param features_file:
    '''
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for f in [vectors_file, entries_file, features_file]:
        shutil.copy(f, output_dir)  # todo this should be a symlink

    final_conf_file = os.path.join(output_dir, os.path.basename(byblo_conf_file))
    shutil.copy(byblo_conf_file, output_dir)
    # restore indices from strings

    thes_prefix = os.path.commonprefix(glob(os.path.join(output_dir, '*strings')))[:-1]
    reindex_all_byblo_vectors(thes_prefix)

    # re-run all-pairs similarity
    # tell byblo to only do the later stages
    set_output_in_byblo_conf_file(final_conf_file, output_dir)

    open(thes_prefix, 'a').close()  # touch this file. Byblo uses the name of the input to find the intermediate files,
    # and complains if the input file does not exist, even if it is not read.

    set_output_in_byblo_conf_file(final_conf_file, thes_prefix, type='input')
    set_stage_in_byblo_conf_file(final_conf_file, 2)
    run_byblo(final_conf_file)
    set_stage_in_byblo_conf_file(final_conf_file, 0)


def baronify_files(entries_file, features_file, vectors_file):
    """
    Removes all entries & vectors that are not contained in a hard-code Baroni thesaurus
    """
    # read the entries
    # todo hardcoded file
    baroni_entries_file = '/mnt/lustre/scratch/inf/mmb28/FeatureExtractionToolkit/exp10-12-composed-ngrams-MR-R2/AN_NN_gigaw-100_Baroni.entries.filtered.strings'
    logging.info('Filtering out entries not in %s', baroni_entries_file)
    with open(baroni_entries_file) as infile:
        baroni_entries = set(line.strip().split()[0] for line in infile)
    logging.info('Accepting %d entries', len(baroni_entries))

    # remove all entries & vectors not included in Baroni whitelist
    new_vectors_file = vectors_file.replace('.events', '.baronified.events')
    with open(vectors_file) as inf, open(new_vectors_file, 'w') as outf:
        for i, line in enumerate(inf):
            if line.split()[0] in baroni_entries:
                outf.write(line)

    new_entries_file = entries_file.replace('.entries', '.baronified.entries')
    with open(entries_file) as inf, open(new_entries_file, 'w') as outf:
        for i, line in enumerate(inf):
            if line.split()[0] in baroni_entries:
                outf.write(line)

    # make features file name consistent with all other file names
    new_features_file = features_file.replace('.features', '.baronified.features')
    if os.path.exists(new_features_file):
        os.unlink(new_features_file)
    os.symlink(features_file, new_features_file)

    return new_entries_file, new_features_file, new_vectors_file


def build_thesauri_out_of_composed_vectors(composer_algos, dataset_name, ngram_vectors_dir,
                                           unigram_thesaurus_dir, baronify):
    byblo_conf_file = _find_conf_file(unigram_thesaurus_dir)
    for c in composer_algos:
        # one phrasal thesaurus per composer
        comp_name = c.name
        vectors_file = os.path.join(ngram_vectors_dir,
                                    'AN_NN_{}_{}.events.filtered.strings'.format(dataset_name, comp_name))
        entries_file = os.path.join(ngram_vectors_dir,
                                    'AN_NN_{}_{}.entries.filtered.strings'.format(dataset_name, comp_name))
        features_file = os.path.join(ngram_vectors_dir,
                                     'AN_NN_{}_{}.features.filtered.strings'.format(dataset_name, comp_name))
        suffix = os.path.basename(vectors_file).split('.')[0]

        if baronify:
            # make a new output dir for this thesaurus
            suffix += '_baronified'
            entries_file, features_file, vectors_file = baronify_files(entries_file, features_file, vectors_file)
        do_second_part_without_base_thesaurus(byblo_conf_file, unigram_thesaurus_dir + suffix,
                                              vectors_file, entries_file, features_file)


def build_unreduced_AN_NN_thesauri(corpus, corpus_name, features,
                                   stages, use_apdt, external_embeddings):
    """
    required files: a Byblo conf file, a labelled classification data set
    created files:  composed vector files in a dir, thesauri of NPs
    """

    # SET UP A FEW REQUIRED PATHS
    byblo_base_dir = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/Byblo-2.2.0/'
    thesisgenerator_base_dir = '/mnt/lustre/scratch/inf/mmb28/thesisgenerator'

    unigram_thesaurus_dir = os.path.abspath(os.path.join(byblo_base_dir, '..',
                                                         'exp%d-%db' % (corpus, features)))  # input 1

    ngram_vectors_dir = os.path.join(byblo_base_dir, '..',
                                     'exp%d-%d-composed-ngrams-MR-R2' % (corpus, features))  # output 1
    # output 2 is a set of directories <output1>*

    apdt_composed_vectors = os.path.join(byblo_base_dir, '..', 'apdt_vectors',
                                         'exp%s-%s_AN_NNvectors-cleaned' % (corpus, features))  # input 3

    if external_embeddings:
        external_events_file = os.path.join(byblo_base_dir, '..',
                                            '%s_vectors' % external_embeddings,
                                            'thesaurus',
                                            '%s.events.filtered.strings' % external_embeddings)
        byblo_conf_external_embeddings = os.path.join(byblo_base_dir, '..',
                                                      '%s_vectors' % external_embeddings,
                                                      '%s.bybloconf.txt' % external_embeddings)

    composer_algos = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer,
                      RightmostWordComposer]  # observed done through a separate script

    if not os.path.exists(ngram_vectors_dir):
        os.mkdir(ngram_vectors_dir)

    # EXTRACT UNIGRAM VECTORS WITH BYBLO
    if 'unigrams' in stages and (not external_embeddings):
        os.chdir(byblo_base_dir)
        calculate_unigram_vectors(unigram_thesaurus_dir)
    else:
        if external_embeddings:
            logging.warn('Skipping unigrams stage. Assuming output is at %s', external_events_file)
        else:
            logging.warn('Skipping unigrams stage. Assuming output is at %s', _find_events_file(unigram_thesaurus_dir))

    # COMPOSE ALL AN/NN VECTORS IN LABELLED SET
    if 'compose' in stages:
        os.chdir(thesisgenerator_base_dir)
        unigram_vectors_file = external_events_file if external_embeddings \
            else _find_events_file(unigram_thesaurus_dir)
        dump.compose_and_write_vectors(unigram_vectors_file,  # I think it is OK for this to contain phrase vectors
                                       corpus_name,
                                       [dump.classification_data_path_mr, dump.classification_data_path],  # input 2
                                       None,
                                       output_dir=ngram_vectors_dir,
                                       composer_classes=composer_algos)
    else:
        logging.warn('Skipping composition stage. Assuming output is at %s', ngram_vectors_dir)

    # BUILD THESAURI OUT OF COMPOSED VECTORS ONLY
    if 'thesauri' in stages:
        os.chdir(byblo_base_dir)
        if use_apdt:
            # convert the pre-composed APDT vectors to Byblo events/entries/features format, adjust paths
            class noop_file(object):
                def write(self, *args, **kwargs):
                    pass

                def __nonzero__(self):
                    return False  # so that we don't get any progress messages from write_vectors_to_disk

                def close(self):
                    pass

            logging.info('Using pre-composed SVD-reduced APDT vectors from %s', apdt_composed_vectors)
            ngram_vectors_dir = os.path.dirname(apdt_composed_vectors)
            composer_algos = [Bunch(name='APDT')]  # only build this thesaurus

            # make a byblo-compatible events, entries and features files
            thesaurus = Vectors.from_tsv([apdt_composed_vectors])
            mat, cols, rows = thesaurus.to_sparse_matrix()

            pattern = os.path.join(ngram_vectors_dir, 'AN_NN_{}_APDT.{}.filtered.strings')
            write_vectors_to_disk(mat.tocoo(),
                                  [DocumentFeature.from_string(x) for x in rows],
                                  cols,
                                  noop_file(),  # no need to write the events file again
                                  pattern.format(corpus_name, 'features'),
                                  pattern.format(corpus_name, 'entries'))

            # symlink to events file so that its name matches that of entries/features
            target_events_file = pattern.format(corpus_name, 'events')
            if os.path.exists(target_events_file):
                os.remove(target_events_file)
            os.symlink(apdt_composed_vectors, target_events_file)

        if external_embeddings and (not os.path.exists(unigram_thesaurus_dir)):
            os.mkdir(unigram_thesaurus_dir)
            shutil.copy(byblo_conf_external_embeddings, unigram_thesaurus_dir)
        build_thesauri_out_of_composed_vectors(composer_algos, corpus_name, ngram_vectors_dir,
                                               unigram_thesaurus_dir, False)  # can't baronify non-reduced thesauri
    else:
        logging.warn('Skipping thesaurus construction stage.')


def build_full_composed_thesauri_with_baroni_and_svd(corpus, features, stages, use_apdt, baronify):
    # SET UP A FEW REQUIRED PATHS

    byblo_base_dir = '/mnt/lustre/scratch/inf/mmb28/FeatureExtractionToolkit/Byblo-2.2.0/'  # trailing slash required
    thesisgenerator_base_dir = '/mnt/lustre/scratch/inf/mmb28/thesisgenerator'
    # INPUT 1:  DIRECTORY. Must contain a single conf file
    unigram_thesaurus_dir = os.path.abspath(os.path.join(byblo_base_dir, '..', 'exp%d-%db' % (corpus, features)))

    # INPUT 2: A FILE, TSV, underscore-separated observed vectors for ANs and NNs
    baroni_training_phrases = os.path.join(byblo_base_dir, '..', 'observed_vectors',
                                           'exp%d-%d_AN_NNvectors-cleaned' % (corpus, features))
    apdt_composed_vectors = os.path.join(byblo_base_dir, '..', 'apdt_vectors',
                                         'exp%d-%d_AN_NNvectors-cleaned' % (corpus, features))

    ngram_vectors_dir = os.path.join(byblo_base_dir, '..',
                                     'exp%d-%d-composed-ngrams-MR-R2' % (corpus, features))  # output 1
    composer_algos = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer,
                      RightmostWordComposer, BaroniComposer]

    target_dimensionality = [100]
    dataset_name = 'gigaw' if corpus == 10 else 'wiki'  # short name of input corpus
    baroni_training_phrase_types = {'AN', 'NN'}  # what kind of NPs to train Baroni composer for

    if not os.path.exists(ngram_vectors_dir):
        os.mkdir(ngram_vectors_dir)

    # EXTRACT UNIGRAM VECTORS WITH BYBLO
    if 'unigrams' in stages:
        os.chdir(byblo_base_dir)
        calculate_unigram_vectors(unigram_thesaurus_dir)
    else:
        logging.warn('Skipping unigrams stage. Assuming output is at %s', _find_events_file(unigram_thesaurus_dir))

    # REDUCE DIMENSIONALITY
    # add in observed AN/NN vectors for SVD processing. Reduce both unigram vectors and observed phrase vectors
    # together and put the output into the same file
    unreduced_unigram_events_file = _find_events_file(unigram_thesaurus_dir)
    # ...exp6-12/exp6.events.filtered.strings --> ...exp6-12/exp6
    reduced_file_prefix = '.'.join(unreduced_unigram_events_file.split('.')[:-3]) + '-with-obs-phrases'
    # only keep the most frequent types per PoS tag to speed things up
    counts = [('N', 20000), ('V', 0), ('J', 10000), ('RB', 00), ('AN', 0), ('NN', 0)]
    if 'svd' in stages:
        if use_apdt:
            reduced_file_prefix = apdt_composed_vectors
            do_svd(unreduced_unigram_events_file, reduced_file_prefix,
                   desired_counts_per_feature_type=counts, reduce_to=target_dimensionality,
                   apply_to=apdt_composed_vectors)  # include both uni- and n-grams vectors in output
        else:
            do_svd(unreduced_unigram_events_file, reduced_file_prefix,
                   desired_counts_per_feature_type=counts, reduce_to=target_dimensionality,
                   apply_to=baroni_training_phrases)
    else:
        logging.warn('Skipping SVD stage. Assuming output is at %s-SVD*', reduced_file_prefix)

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

        if 'baroni' in stages:
            # do the actual training
            thes = Vectors.from_tsv([all_reduced_vectors], lowercasing=False)
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
            logging.warn('Skipping Baroni training stage. Assuming trained models are at %s', trained_composer_files)

    if 'compose' in stages and not use_apdt:
        # not point in composing with APTD, composed vectors are already there for me
        os.chdir(thesisgenerator_base_dir)
        for svd_dims, all_reduced_vectors, trained_composer_file \
                in zip(target_dimensionality, baroni_training_data, trained_composer_files):
            dump.compose_and_write_vectors(all_reduced_vectors,
                                           '%s-%s' % (dataset_name, svd_dims),
                                           [dump.classification_data_path_mr, dump.classification_data_path],
                                           trained_composer_file,
                                           output_dir=ngram_vectors_dir,
                                           composer_classes=composer_algos)
    else:
        logging.warn('Skipping composition stage. Assuming output is at %s', ngram_vectors_dir)

    # BUILD THESAURI OUT OF COMPOSED VECTORS ONLY
    for dims in target_dimensionality:
        if 'thesauri' in stages:
            os.chdir(byblo_base_dir)
            tmp_dataset_name = '%s-%d' % (dataset_name, dims)
            if use_apdt:
                # filenames do not match because we didn't use compose_and_write_vectors
                # throw in a few symlinks to fix that
                logging.info('Using pre-composed SVD-reduced APDT vectors from %s', apdt_composed_vectors)
                ngram_vectors_dir = os.path.dirname(apdt_composed_vectors)
                composer_algos = [Bunch(name='APDT')]  # only build this thesaurus
                for x in ['events', 'entries', 'features']:
                    # AN_NN_gigaw-30_APDT.events.filtered.strings, what Byblo expects
                    expected_vectors_file = os.path.join(ngram_vectors_dir,
                                                         'AN_NN_{}_APDT.{}.filtered.strings'.format(tmp_dataset_name,
                                                                                                    x))
                    # exp10-12_AN_NNvectors-SVD30.events.filtered.strings, what the SVD stage produced
                    actual_file = os.path.join(ngram_vectors_dir,
                                               'exp{}-{}_AN_NNvectors-cleaned-SVD{}.{}.filtered.strings'.format(corpus,
                                                                                                                features,
                                                                                                                dims,
                                                                                                                x))
                    if not os.path.exists(actual_file):
                        raise ValueError('File %s not found' % actual_file)
                    if not os.path.exists(expected_vectors_file):
                        os.symlink(actual_file, expected_vectors_file)

            build_thesauri_out_of_composed_vectors(composer_algos, tmp_dataset_name,
                                                   ngram_vectors_dir, unigram_thesaurus_dir, baronify)
        else:
            logging.warn('Skipping thesaurus construction stage. Assuming output is at %s', ngram_vectors_dir)


def get_corpus_features_cmd_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--corpus', choices=('wikipedia', 'gigaword', 'neuro', 'word2vec'), required=True,
                        help='Unlabelled corpus to source unigram vectors from')
    parser.add_argument('--features', choices=('dependencies', 'windows', 'neuro', 'word2vec'), required=True,
                        help='Feature type of unigram vectors')
    parser.add_argument('--baronify', action='store_true', required=False,
                        help='If true, only entries that Baroni can compose will be included in thesauri')
    return parser


def get_cmd_parser():
    parser = argparse.ArgumentParser(parents=[get_corpus_features_cmd_parser()])
    # add options specific to this script here
    parser.add_argument('--stages', choices=('unigrams', 'svd', 'baroni', 'compose', 'thesauri'), required=True,
                        nargs='+',
                        help='What parts of the pipeline to run. Each part is independent, the pipeline can be '
                             'run incrementally. The stages are: '
                             '1) unigrams: extract unigram vectors from unlabelled corpus '
                             '2) svd: reduce noun and adj matrix, apply to NP matrix '
                             '3) baroni: train Baroni composer '
                             '4) compose: compose all possible NP vectors with all composers '
                             '5) thesauri: build thesauri from available composed vectors')
    parser.add_argument('--use-svd', action='store_true',
                        help='If set, SVD will be performed and a Baroni composer will be trained. Otherwise the'
                             'svd part of the pipeline is skipped.')

    parser.add_argument('--use-apdt', action='store_true',
                        help='If set, APDT pre-composed vectors will be used instead of all other composition methods.')

    parser.add_argument('--external-embeddings', choices=('socher', 'word2vec', None), default=None,
                        help='If set, a set of external (usually neural) embedings will be used as unigram vectors '
                             'instead of wiki/gigaw count-based ones. These must already be in a predefined location.'
                             'Only Add/Mult/Left/Right composers work with this option. SVD must not be enabled as '
                             'these embedding are already low-dimensional. The corpus and features arguments are '
                             'ignored.')

    return parser


if __name__ == '__main__':
    import sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %(message)s")

    parameters = get_cmd_parser().parse_args()
    logging.info(parameters)
    if parameters.features != 'dependencies' and parameters.use_apdt:
        raise ValueError('APDT only works with dependency features!')

    corpus_name = parameters.corpus[:5]  # human-readable corpus name
    # here 'neuro' stands for Socher(2011) for historical reasons
    corpus = {'gigaword': 10, 'wikipedia': 11, 'neuro': 12, 'word2vec': 13}
    features = {'dependencies': 12, 'windows': 13, 'neuro': 14, 'word2vec': 15}

    if parameters.use_svd:
        logging.info('Starting pipeline with SVD and Baroni composer')
        build_full_composed_thesauri_with_baroni_and_svd(corpus[parameters.corpus], features[parameters.features],
                                                         parameters.stages, parameters.use_apdt, parameters.baronify)
    else:
        logging.info('Starting pipeline without SVD')
        build_unreduced_AN_NN_thesauri(corpus[parameters.corpus], corpus_name, features[parameters.features],
                                       parameters.stages, parameters.use_apdt, parameters.external_embeddings)
