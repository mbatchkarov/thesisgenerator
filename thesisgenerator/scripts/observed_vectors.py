import sys

sys.path.append('.')
import argparse
import logging
from os.path import join, dirname
from discoutils.cmd_utils import run_and_log_output
from discoutils.misc import temp_chdir, mkdirs_if_not_exists
from discoutils.thesaurus_loader import Vectors
from discoutils.reweighting import ppmi_sparse_matrix
from thesisgenerator.scripts.extract_NPs_from_labelled_data import NP_MODIFIERS_FILE, VERBS_FILE

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

    fet_output = 'exp10' if args.corpus == 'gigaw' else 'exp11'  # the output of Feature Extraction Toolkit
    prefix = '/mnt/lustre/scratch/inf/mmb28/'
    TG = join(prefix, 'thesisgenerator')
    byblo_base_dir = join(prefix, 'FeatureExtractionToolkit/', 'Byblo-2.2.0')
    discoutils = join(prefix, 'DiscoUtils')
    byblo_filter_thresholds = 50, 50, 10
    min_corpus_freq, min_features = 10, 9 # num_fields_in_line = 1 + 2*num_features

    # Find all NPs in unlabelled corpus whose modifier is in labelled corpus,
    # and which appear more than thresh=100 times in unlabelled corpus
    script = join(discoutils, 'discoutils', 'find_all_NPs.py')
    byblo_features_file = join(prefix, 'FeatureExtractionToolkit',
                               'feoutput-deppars', fet_output)
    out1 = join(discoutils, '%s_NPs_in_MR_R2_TechTC_am_maas.txt' % args.corpus)
    out2 = join(discoutils, '%s_NPs_in_MR_R2_TechTC_am_maas.uniq.%d.txt' % (args.corpus, min_corpus_freq))
    out3 = join(discoutils,
                '%s-obs-wins' % args.corpus,
                '%s-obs-wins.fet' % args.corpus)

    with temp_chdir(discoutils):
        # find all NPs in UNLABELLED corpus, whose modifier appears in the LABELLED corpora
        run_and_log_output('python {} {} --output {} --whitelist {} {}',
                           script, byblo_features_file, out1,
                           join(TG, NP_MODIFIERS_FILE), join(TG, VERBS_FILE))

        # filter out those that occur infrequently in unlab data
        run_and_log_output("cat {} | sort | uniq -c | awk '$1>{} {print $2}' > {}",
                           out1, str(min_corpus_freq), out2)

        mkdirs_if_not_exists(join(discoutils,
                                  '%s-obs-wins' % args.corpus))

        # go through events file again and print features for the NPs that survive filtering
        run_and_log_output('python discoutils/find_all_NPs.py {} --vectors --whitelist {} --output {}',
                           byblo_features_file, out2, out3)
        # the whitelist needs to contain the verbs

    with temp_chdir(byblo_base_dir):
        # collect features from all occurences of an NP into a single vector
        run_and_log_output('./byblo.sh -i {} -o {} -t 10 --stages enumerate,count,filter '
                           '--filter-entry-freq {} --filter-feature-freq {} --filter-event-freq {}',
                           out3, dirname(out3), *map(str, byblo_filter_thresholds))
        run_and_log_output('./unindex-all.sh {}', out3)

    # remove NPs with too few features
    obs_vectors_dir = join(prefix, 'FeatureExtractionToolkit', 'observed_vectors')
    output_path = join(obs_vectors_dir, '%s_NPs_wins_observed' % args.corpus)
    mkdirs_if_not_exists(obs_vectors_dir)
    with temp_chdir(dirname(out3)):
        run_and_log_output("awk 'NF>{}' {}-obs-wins.fet.events.filtered.strings >  {}",
                           str(min_features), args.corpus,
                           output_path)

    # now do PPMI
    # todo this should ideally be done together with unigram vectors
    v = Vectors.from_tsv(output_path)
    ppmi_sparse_matrix(v.matrix)
    v.to_tsv(output_path + '_ppmi', gzipped=True)
