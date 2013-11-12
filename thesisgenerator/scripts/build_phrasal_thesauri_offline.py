import logging
import os
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
from glob import glob
from thesisgenerator.plugins.thesaurus_loader import Thesaurus
from thesisgenerator.utils.cmd_utils import set_stage_in_byblo_conf_file, run_byblo, parse_byblo_conf_file, \
    reindex_all_byblo_vectors, run_and_log_output, unindex_all_byblo_vectors
from thesisgenerator.scripts import dump_all_composed_vectors as dump


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


def _find_output_prefix(thesaurus_dir):
    return os.path.commonprefix(glob(os.path.join(thesaurus_dir, '*filtered*')))[:-1]


def do_second_part(thesaurus_dir, feature_type=None):
    thes_prefix = _find_output_prefix(thesaurus_dir)
    byblo_conf_file = _find_conf_file(thesaurus_dir)

    if feature_type:
        tweaked_vector_files = _find_new_files(feature_type, ngram_vectors_dir)
        add_new_vectors(thesaurus_dir, *tweaked_vector_files)
        # restore indices from strings
        reindex_all_byblo_vectors(thes_prefix)

    # re-run all-pairs similarity
    set_stage_in_byblo_conf_file(byblo_conf_file, 2)
    run_byblo(byblo_conf_file)
    set_stage_in_byblo_conf_file(byblo_conf_file, 0)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s ""(line %(lineno)d)\t%(levelname)s : %(""message)s")

    byblo_base_dir = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/Byblo-2.2.0/' # trailing slash required

    thesaurus_dirs = [
        os.path.join(byblo_base_dir, '..', 'exp6-12%s' % x) for x in 'abcd'
    ]

    ngram_vectors_dir = os.path.join(byblo_base_dir, '..', 'exp6-12-ngrams')
    if not os.path.exists(ngram_vectors_dir):
        os.mkdir(ngram_vectors_dir)

    os.chdir(byblo_base_dir)
    for thesaurus_dir in thesaurus_dirs:
        calculate_unigram_vectors(thesaurus_dir)

    # mess with vectors, add to/modify entries and events files
    # whether to modify the features file is less obvious- do composed entries have different features
    # to the non-composed ones?
    tweaked_vector_files = [
        os.path.join(byblo_base_dir, 'sample-data', 'output', 'bigram_7head_bar_svo.vectors.tsv'),
        os.path.join(byblo_base_dir, 'sample-data', 'output', 'bigram_7head_bar_svo.entries.txt'),
        os.path.join(byblo_base_dir, 'sample-data', 'output', 'bigram_7head_bar_svo.features.txt')]

    event_files = [glob(os.path.join(dir, '*events.filtered.strings'))[0] for dir in thesaurus_dirs]
    dump.write_vectors(event_files,
                       dump.data_path,
                       log_to_console=True,
                       output_dir=ngram_vectors_dir)

    # add AN phrases to noun thesaurus, SVO to verb thesaurus, and rebuild
    do_second_part(thesaurus_dirs[0], feature_type='AN') # nouns
    do_second_part(thesaurus_dirs[1], feature_type='SVO') # verbs
    #do_second_part('VO', thesaurus_dirs[1])
    do_second_part(thesaurus_dirs[2]) # adjectives
    do_second_part(thesaurus_dirs[3]) # adverbs

    thesaurus = Thesaurus(['{}.sims.neighbours.strings'.format(_find_output_prefix(thesaurus_dirs[0]))])
    print thesaurus.get('expand/V force/N')
    print thesaurus.get('military/J force/N')
    print thesaurus.get('thursday/N')

    thesaurus = Thesaurus(['{}.sims.neighbours.strings'.format(_find_output_prefix(thesaurus_dirs[1]))])
    print thesaurus.get('think/V')
    print thesaurus.get('center/N sign/V agreement/N')