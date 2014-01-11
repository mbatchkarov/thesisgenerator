import logging
import os
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
import re
from glob import glob
import shutil
from thesisgenerator.plugins.thesaurus_loader import Thesaurus
from thesisgenerator.utils.cmd_utils import set_stage_in_byblo_conf_file, run_byblo, parse_byblo_conf_file, \
    reindex_all_byblo_vectors, run_and_log_output, unindex_all_byblo_vectors, set_output_in_byblo_conf_file
from thesisgenerator.scripts import dump_all_composed_vectors as dump
from thesisgenerator.scripts.reduce_dimensionality import do_svd
from thesisgenerator.composers.utils import reformat_entries, julie_transform, julie_transform2
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

    with open(features_file, 'a+b') as outfile: # append mode
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
            shutil.rmtree(copy_to_dir) # copytree will fail if target exists
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
    Takes a set of plain-text TSB files and builds a thesaurus out of them
    :param byblo_conf_file: File that specifies filtering, similarity measure, etc
    :param output_dir: where should the output go
    :param vectors_file:
    :param entries_file:
    :param features_file:
    '''
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    all_files = [vectors_file, entries_file, features_file]
    for f in all_files:
        shutil.copy(f, output_dir)

    final_conf_file = os.path.join(output_dir, os.path.basename(byblo_conf_file))
    shutil.copy(byblo_conf_file, output_dir)
    # restore indices from strings
    thes_prefix = _find_output_prefix(output_dir)
    reindex_all_byblo_vectors(thes_prefix)

    # re-run all-pairs similarity
    # tell byblo to only do the later stages
    set_output_in_byblo_conf_file(final_conf_file, output_dir)

    open(thes_prefix, 'a').close() # touch this file. Byblo uses the name of the input to find the intermediate files,
    # and complains if the input file does not exist, even if it is not read.

    set_output_in_byblo_conf_file(final_conf_file, thes_prefix, type='input')
    set_stage_in_byblo_conf_file(final_conf_file, 2)
    run_byblo(final_conf_file)
    set_stage_in_byblo_conf_file(final_conf_file, 0)


def build_only_AN_NN_thesauri_without_baroni(exp):
    # required files: a Byblo conf file, a labelled classification data set
    # created files:  composed vector files in a dir, thesauri of NPs

    # SET UP A FEW REQUIRED PATHS
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s ""(line %(lineno)d)\t%(levelname)s : %(""message)s")

    byblo_base_dir = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/Byblo-2.2.0/'

    unigram_thesaurus_dir = os.path.abspath(os.path.join(byblo_base_dir, '..', 'exp%d-12b' % exp)) # todo input 1

    ngram_vectors_dir = os.path.join(byblo_base_dir, '..', 'exp%d-12-composed-ngrams-MR-R2' % exp) # output 1
    # output 2 is a set of directories <output1>*

    composer_algos = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer,
                      RightmostWordComposer, MinComposer, MaxComposer] # observed done through a separate script

    # EXTRACT UNIGRAM VECTORS WITH BYBLO
    if not os.path.exists(ngram_vectors_dir):
        os.mkdir(ngram_vectors_dir)
    os.chdir(byblo_base_dir)
    calculate_unigram_vectors(unigram_thesaurus_dir)

    # COMPOSE ALL AN/NN VECTORS IN LABELLED SET
    unigram_vectors_file = _find_events_file(unigram_thesaurus_dir)
    dump.compose_and_write_vectors([unigram_vectors_file],
                                   'gigaw', # todo short name of input 2
                                   [dump.classification_data_path_mr, dump.classification_data_path], # todo input 2
                                   None,
                                   output_dir=ngram_vectors_dir,
                                   composer_classes=composer_algos)

    # BUILD THESAURI OUT OF COMPOSED VECTORS ONLY
    byblo_conf_file = _find_conf_file(unigram_thesaurus_dir)
    for c in composer_algos:
        # one phrasal thesaurus per composer
        name = c.name
        vectors_file = os.path.join(ngram_vectors_dir, 'AN_NN_gigaw_{}.events.filtered.strings'.format(name))
        entries_file = os.path.join(ngram_vectors_dir, 'AN_NN_gigaw_{}.entries.filtered.strings'.format(name))
        features_file = os.path.join(ngram_vectors_dir, 'AN_NN_gigaw_{}.features.filtered.strings'.format(name))
        suffix = os.path.basename(vectors_file).split('.')[0]
        do_second_part_without_base_thesaurus(byblo_conf_file, unigram_thesaurus_dir + suffix,
                                              vectors_file, entries_file, features_file)


def build_full_composed_thesauri_with_baroni_and_svd():
    global byblo_base_dir, thesaurus_dirs, ngram_vectors_dir, composer_algos, dir, files_to_reduce, x, reduced_prefixes, baroni_training_phrase_types, reduce_to, counts, prefix, dims, pref, all_vectors, svd_settings, thes, trained_composers, event_files, source, c, name, vectors_file, entries_file, features_file, suffix, thesaurus, entry
    # SET UP A FEW REQUIRED PATHS
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s ""(line %(lineno)d)\t%(levelname)s : %(""message)s")
    byblo_base_dir = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/Byblo-2.2.0/' # trailing slash required
    #thesaurus_dirs = [
    #    os.path.abspath(os.path.join(byblo_base_dir, '..', 'exp6-12%s' % x)) for x in 'abcd'
    #]
    thesaurus_dirs = [
        os.path.abspath(os.path.join(byblo_base_dir, '..', 'exp10-12b'))
    ]
    ngram_vectors_dir = os.path.join(byblo_base_dir, '..', 'exp10-12-ngrams-MR')
    composer_algos = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer,
                      RightmostWordComposer, MinComposer, MaxComposer] # todo add ['observed'] here
    # EXTRACT UNIGRAM VECTORS WITH BYBLO
    if not os.path.exists(ngram_vectors_dir):
        os.mkdir(ngram_vectors_dir)
    os.chdir(byblo_base_dir)
    # for thesaurus_dir in thesaurus_dirs:
    #     calculate_unigram_vectors(thesaurus_dir)
    # REDUCE DIMENSIONALITY
    files_to_reduce = [_find_events_file(dir) for dir in thesaurus_dirs]
    # output write everything to the same directory
    # ...exp6-12/exp6.events.filtered.strings --> ...exp6-12/exp6
    reduced_prefixes = ['.'.join(x.split('.')[:-3]) + '-with-obs-phrases' for x in files_to_reduce]
    # obtain training data for composer, that also needs to be reduced
    baroni_training_phrase_types = ['AN', 'NN']
    #baroni_training_phrases = [os.path.abspath(os.path.join(byblo_base_dir, '..', 'phrases',
    #                                                        'julie.{}s.vectors'.format(x)))
    #                           for x in baroni_training_phrase_types]
    # convert from Julie's format to mine
    # convert to dissect format (underscore-separated ANs) for composer training
    #baroni_training_phrases.append(reformat_entries(baroni_training_phrases[0], 'clean',
    #                                                function=lambda x: julie_transform(x, separator='_')))
    #baroni_training_phrases.append(reformat_entries(baroni_training_phrases[1], 'clean',
    #                                                function=lambda x: julie_transform2(x, separator='_', pos1='N')))
    # add in observed AN/NN vectors for SVD processing
    #files_to_reduce.extend(baroni_training_phrases)
    if False:        # DO NOT DO ANY SVD FOR NOW
        reduce_to = [300, 500]
        counts = [('N', 8000), ('V', 4000), ('J', 4000), ('RB', 200), ('AN', 20000), ('NN', 20000)]
        do_svd(files_to_reduce, reduced_prefixes, desired_counts_per_feature_type=counts, reduce_to=reduce_to)

        reduced_prefixes = ['%s-SVD%d' % (prefix, dims) for prefix in reduced_prefixes for dims in reduce_to]
    else:
        # look at the original file paths
        reduced_prefixes = ['.'.join(x.split('.')[:-3]) for x in files_to_reduce]

    # TRAIN BARONI COMPOSER
    # train on each SVD-reduced file, not the original one
    for pref in reduced_prefixes:
        # find file and its svd dimensionality from prefix
        all_vectors = pref + '.events.filtered.strings'
        try:
            svd_settings = re.search(r'.*(SVD[0-9]+).*', pref).group(1)
        except AttributeError:
            # 'NoneType' object has no attribute 'group'
            svd_settings = ''
            # load it and extract just the nouns/ ANs to train Baroni composer on
        thes = Thesaurus.from_tsv([all_vectors], aggressive_lowercasing=False)

        trained_composers = []
        #for training_phrases, phrase_type in zip(baroni_training_phrases, baroni_training_phrase_types):
        #    baroni_training_heads = training_phrases.replace(phrase_type, 'onlyN-%s' % svd_settings)
        #    thes.to_file(baroni_training_heads,
        #                 entry_filter=lambda x: x.type == '1-GRAM' and x.tokens[0].pos == 'N')
        #
        #    baroni_training_only_phrases = training_phrases.replace(phrase_type,
        #                                                            'only%s-%s' % (phrase_type, svd_settings))
        #    thes.to_file(baroni_training_only_phrases,
        #                 entry_filter=lambda x: x.type == phrase_type,
        #                 row_transform=lambda x: x.replace(' ', '_'))
        #
        #    baroni_trained_model_output_prefix = training_phrases.replace('vectors',
        #                                                                  '%s-model-%s' % ( phrase_type, svd_settings))
        #    trained_composer_path = baroni_trained_model_output_prefix + '.model.pkl'
        #    trained_composer_path = train_baroni_composer(baroni_training_heads,
        #                                                  baroni_training_only_phrases,
        #                                                  baroni_trained_model_output_prefix)
        #    trained_composers.append(trained_composer_path)

        # mess with vectors, add to/modify entries and events files
        # whether to modify the features file is less obvious- do composed entries have different features
        # to the non-composed ones?
        event_files = [_find_events_file(dir) for dir in thesaurus_dirs]

        dump.compose_and_write_vectors([all_vectors],
                                       'wiki-%s' % svd_settings if svd_settings else 'wiki',
                                       dump.classification_data_path_mr,
                                       trained_composers,
                                       output_dir=ngram_vectors_dir,
                                       composer_classes=composer_algos)
    source = thesaurus_dirs[0]
    print source
    # do_second_part2(source) #original unigram-only thesaurus
    for c in composer_algos:
        # one phrasal thesaurus per composer
        name = c.name
        vectors_file = os.path.join(ngram_vectors_dir, 'AN_NN_wiki_{}.vectors.tsv'.format(name))
        entries_file = os.path.join(ngram_vectors_dir, 'AN_NN_wiki_{}.entries.txt'.format(name))
        features_file = os.path.join(ngram_vectors_dir, 'AN_NN_wiki_{}.features.txt'.format(name))
        suffix = os.path.basename(vectors_file).split('.')[0]
        do_second_part(source, vectors_file, entries_file, features_file, copy_to_dir=source + suffix)
    sys.exit(0) #ENOUGH FOR NOW
    do_second_part(thesaurus_dirs[0], add_feature_type=['AN', 'NN']) #'VO', 'SVO' # all vectors in same thesaurus
    do_second_part(thesaurus_dirs[0]) # plain old thesaurus without ngrams
    ### add AN phrases to noun thesaurus, SVO to verb thesaurus, and rebuild
    #do_second_part(thesaurus_dirs[0], add_feature_type=['AN']) # nouns with ngrams
    #do_second_part(thesaurus_dirs[0]) # nouns
    #do_second_part(thesaurus_dirs[1]) # verbs
    #do_second_part(thesaurus_dirs[1], add_feature_type=['SVO']) # verbs with ngrams
    ##do_second_part('VO', thesaurus_dirs[1])
    #do_second_part(thesaurus_dirs[2]) # adjectives
    #do_second_part(thesaurus_dirs[3]) # adverbs
    for thesaurus in [
        Thesaurus([_find_allpairs_file(thesaurus_dirs[0])]),
        Thesaurus([_find_allpairs_file(thesaurus_dirs[0] + '-with-ngrams')])]:

        for entry in ['thursday/N', 'expand/V force/N', 'military/J force/N',
                      'center/N sign/V agreement/N', 'think/V']:
            print entry, '------->', thesaurus.get(entry)
        print '--------------------'


if __name__ == '__main__':
    import sys

    build_only_AN_NN_thesauri_without_baroni(int(sys.argv[1]))
    # build_full_composed_thesauri_with_baroni_and_svd()
