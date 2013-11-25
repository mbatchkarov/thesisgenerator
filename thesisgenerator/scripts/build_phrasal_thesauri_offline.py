import logging
import os
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
from glob import glob
from shutil import copytree, rmtree
from thesisgenerator.plugins.thesaurus_loader import Thesaurus
from thesisgenerator.utils.cmd_utils import set_stage_in_byblo_conf_file, run_byblo, parse_byblo_conf_file, \
    reindex_all_byblo_vectors, run_and_log_output, unindex_all_byblo_vectors, set_output_in_byblo_conf_file
from thesisgenerator.scripts import dump_all_composed_vectors as dump
from thesisgenerator.scripts.reduce_dimensionality import do_work


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


def do_second_part(thesaurus_dir, add_feature_type=[]):
    if add_feature_type:
        # if entries are to be added, make a copy of the entire output of the first stage so
        # that the unmodified thesaurus can still be built
        new_thes_dir = thesaurus_dir + '-with-ngrams'
        if os.path.exists(new_thes_dir):
            rmtree(new_thes_dir) # copytree will fail if target exists
        copytree(thesaurus_dir, new_thes_dir)
        thesaurus_dir = new_thes_dir

        for feature_type in add_feature_type:
            tweaked_vector_files = _find_new_files(feature_type, ngram_vectors_dir)
            add_new_vectors(new_thes_dir, *tweaked_vector_files)

        # restore indices from strings
        thes_prefix = _find_output_prefix(new_thes_dir)
        reindex_all_byblo_vectors(thes_prefix)

    # re-run all-pairs similarity
    # first change output prefix in conf file, in case the if-statement above has made a new working directory
    byblo_conf_file = _find_conf_file(thesaurus_dir)
    # tell byblo to only do the later stages
    set_output_in_byblo_conf_file(byblo_conf_file, thesaurus_dir)
    set_stage_in_byblo_conf_file(byblo_conf_file, 2)
    run_byblo(byblo_conf_file)
    set_stage_in_byblo_conf_file(byblo_conf_file, 0)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s ""(line %(lineno)d)\t%(levelname)s : %(""message)s")

    byblo_base_dir = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/Byblo-2.2.0/' # trailing slash required

    #thesaurus_dirs = [
    #    os.path.abspath(os.path.join(byblo_base_dir, '..', 'exp6-12%s' % x)) for x in 'abcd'
    #]
    thesaurus_dirs = [
        os.path.abspath(os.path.join(byblo_base_dir, '..', 'exp6-12'))
    ]

    ngram_vectors_dir = os.path.join(byblo_base_dir, '..', 'exp6-12-ngrams')

    if not os.path.exists(ngram_vectors_dir):
        os.mkdir(ngram_vectors_dir)
    os.chdir(byblo_base_dir)

    #for thesaurus_dir in thesaurus_dirs:
    #    calculate_unigram_vectors(thesaurus_dir)

    # reduce dimensionality
    do_work([_find_events_file(dir) for dir in thesaurus_dirs], reduce_to=[3, 5])
    sys.exit(0) # ENOUGH FOR NOW

    # mess with vectors, add to/modify entries and events files
    # whether to modify the features file is less obvious- do composed entries have different features
    # to the non-composed ones?
    event_files = [_find_events_file(dir) for dir in thesaurus_dirs]
    dump.write_vectors(event_files,
                       dump.data_path,
                       log_to_console=True,
                       output_dir=ngram_vectors_dir)

    do_second_part(thesaurus_dirs[0], add_feature_type=['AN', 'VO', 'SVO']) # all vectors in same thesaurus
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