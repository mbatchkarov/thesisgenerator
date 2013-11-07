import logging
import os
from thesisgenerator.plugins.thesaurus_loader import Thesaurus
from thesisgenerator.utils.cmd_utils import set_stage_in_byblo_conf_file, run_byblo, parse_byblo_conf_file, \
    reindex_all_byblo_vectors, run_and_log_output, unindex_all_byblo_vectors
import thesisgenerator.scripts.dump_all_composed_vectors as dump


def calculate_unigram_vectors(byblo_output_prefix, byblo_conf_file):
    # calculate vectors for all entries
    set_stage_in_byblo_conf_file(byblo_conf_file, 1)
    run_byblo(byblo_conf_file)
    set_stage_in_byblo_conf_file(byblo_conf_file, 0)
    # get vectors as strings
    unindex_all_byblo_vectors(byblo_output_prefix)


def add_new_vectors(byblo_output_prefix, composed_vectors, composed_entries, composed_features):
    # todo this appending should only occur once, should I check for it?
    run_and_log_output('cat {} >> {}.entries.filtered.strings'.format(composed_entries, byblo_output_prefix))
    run_and_log_output('cat {} >> {}.events.filtered.strings'.format(composed_vectors, byblo_output_prefix))

    # features of newly created entries must be the same as these of the old ones, but let's check just in case
    # Byblo throws an out-of-bounds exception if features do not match
    with open('{}.features.filtered.strings'.format(byblo_output_prefix)) as infile:
        old_features = set(x.strip().split('\t')[0] for x in infile.readlines())
    with open(composed_features) as infile:
        new_features = [x.strip().split('\t') for x in infile.readlines()]

    with open('{}.features.filtered.strings'.format(byblo_output_prefix), 'a+b') as outfile:
        for feature, count in new_features:
            if feature not in old_features:
                outfile.write('%s\t%s\n' % (feature, count))
            else:
                logging.debug('Ignoring duplicate feature %s with count %s', feature, count)


def rebuild_thesaurus(byblo_output_prefix, byblo_conf_file):
    # restore indices from strings
    reindex_all_byblo_vectors(byblo_output_prefix)

    # re-run all-pairs similarity
    set_stage_in_byblo_conf_file(byblo_conf_file, 2)

    run_byblo(byblo_conf_file)
    set_stage_in_byblo_conf_file(byblo_conf_file, 0)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s ""(line %(lineno)d)\t%(levelname)s : %(""message)s")

    byblo_base_dir = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/Byblo-2.2.0'
    os.chdir(byblo_base_dir)
    byblo_conf_file = os.path.join(byblo_base_dir, 'sample-data', 'conf.txt')

    opts, _ = parse_byblo_conf_file(byblo_conf_file)
    byblo_output_prefix = '{}{}'.format(opts.output, os.path.basename(opts.input))

    calculate_unigram_vectors(byblo_output_prefix, byblo_conf_file)

    # mess with vectors, add to/modify entries and events files
    # whether to modify the features file is less obvious- do composed entries have different features
    # to the non-composed ones?
    tweaked_vector_files = [
        os.path.join(byblo_base_dir, 'sample-data', 'output', 'bigram_7head_bar_svo.vectors.tsv'),
        os.path.join(byblo_base_dir, 'sample-data', 'output', 'bigram_7head_bar_svo.entries.txt'),
        os.path.join(byblo_base_dir, 'sample-data', 'output', 'bigram_7head_bar_svo.features.txt')]

    tweaked_vector_files = dump.write_vectors(['{}.events.filtered.strings'.format(byblo_output_prefix)],
                                              map(lambda x: '%s-small' % x, dump.data_path),
                                              log_to_console=True,
                                              output_dir=opts.output)
    add_new_vectors(byblo_output_prefix, *tweaked_vector_files)
    rebuild_thesaurus(byblo_output_prefix, byblo_conf_file)

    thesaurus = Thesaurus(['{}.sims.neighbours.strings'.format(byblo_output_prefix)])
    print thesaurus.get('final/J week/N')
    print thesaurus.get('fame/N')