import argparse
import logging
from os.path import join, dirname
from discoutils.cmd_utils import run_and_log_output
from discoutils.misc import temp_chdir, mkdirs_if_not_exists

"""
Implements the Evernote note "Training Baroni composers for NP experiments"
Jumps throught a bunch of hoops to extract observed window vectors for NPs
List of phrases in labelled corpus must have been extracted already
 """
# this produces a bunch of all temporary files. Should they be marked as such?
# not all steps steps handle empty inputs well
# if any crashes with a weird error, it's likely the filtering thresholds are too high


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', choices=('gigaw', 'wiki'), required=True)
    args = parser.parse_args()

    features_file = 'exp10' if args.corpus == 'gigaw' else 'exp11'  # the output of FET
    prefix = '/mnt/lustre/scratch/inf/mmb28/'
    byblo_base_dir = join(prefix, 'FeatureExtrationToolkit/', 'Byblo-2.2.0')
    discoutils = join(prefix, 'DiscoUtils')
    modifiers = join(prefix, 'thesisgenerator',
                     'NPs_in_R2_MR_tech_am_maas',
                     'r2-mr-technion-am-maas-modifiers.txt')
    byblo_filter_thresholds = 200, 200, 100
    min_corpus_freq, min_features = 100, 10

    # Find all NPs in unlabelled corpus whose modifier is in labelled corpus,
    # and which appear more than thresh=100 times in unlabelled corpus
    script = join(discoutils, 'discoutils', 'find_all_NPs.py')
    byblo_features_file = join(prefix, 'FeatureExtrationToolkit',
                               'feoutput-deppars', features_file)
    out1 = join(discoutils, '%s_NPs_in_MR_R2_TechTC_am_maas.txt' % args.corpus)
    out2 = join(discoutils, '%s_NPs_in_MR_R2_TechTC_am_maas.uniq.%d.txt' % (args.corpus, min_corpus_freq))
    out3 = join(discoutils,
                '%s-obs-wins' % args.corpus,
                '%s-obs-wins.fet' % args.corpus)

    with temp_chdir(discoutils):
        cmd = 'python {} {} -o {} -s {}'
        run_and_log_output(cmd, script, byblo_features_file, out1, modifiers)

        run_and_log_output("cat {} | sort | uniq -c | awk '$1>{} {print $2}' > {}",
                           out1, str(min_corpus_freq), out2)

        mkdirs_if_not_exists(join(discoutils,
                                  '%s-obs-wins' % args.corpus))
        run_and_log_output('python discoutils/find_all_NPs.py {} -v -s {} -o {}',
                           byblo_features_file, out2, out3)

    with temp_chdir(byblo_base_dir):
        run_and_log_output('./byblo.sh -i {} -o {} -t 10 --stages enumerate,count,filter '
                           '--filter-entry-freq {} --filter-feature-freq {} --filter-event-freq {}',
                           out3, dirname(out3), *map(str, byblo_filter_thresholds))
        run_and_log_output('./unindex-all.sh {}', out3)

    with temp_chdir(dirname(out3)):
        obs_vectors_dir = join(prefix, 'FeatureExtractionToolkit', 'observed_vectors')
        mkdirs_if_not_exists(obs_vectors_dir)
        run_and_log_output("awk 'NF>{}' {}-obs-wins.fet.events.filtered.strings >  {}",
                           str(min_features), args.corpus,
                           join(obs_vectors_dir, '%s_NPs_wins_observed' % args.corpus))