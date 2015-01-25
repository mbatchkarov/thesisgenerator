from os.path import join, dirname
from discoutils.cmd_utils import run_and_log_output
from discoutils.misc import temp_chdir, mkdirs_if_not_exists

"""
Implements the Evernote note "Training Baroni composers for NP experiments"
Jumps throught a bunch of hoops to extract observed window vectors for NPs
List of phrases in labelled corpus must have been extracted already
 """
# todo this contains a number of hardcoded thresholds
# todo all temporary paths should be marked as such
# todo final output file naming is a mess to comply with old convention, could be neater
# todo need to parameterize for gigaw/wikipedia

prefix = '/mnt/lustre/scratch/inf/mmb28/'
byblo_base_dir = join(prefix, 'FeatureExtrationToolkit/', 'Byblo-2.2.0')
discoutils = join(prefix, 'DiscoUtils')
modifiers = join(prefix, 'thesisgenerator',
                         'NPs_in_R2_MR_tech_am_maas',
                         'r2-mr-technion-am-maas-modifiers.txt')

# Find all NPs in unlabelled corpus whose modifier is in labelled corpus,
# and which appear more than thresh=100 times in unlabelled corpus
thresh = '100'  # this needs to be a str, otherwise iterpipes breaks
script = join(discoutils, 'discoutils', 'find_all_NPs.py')
byblo_features_file = join(prefix, 'FeatureExtrationToolkit',
                                   'feoutput-deppars', 'exp10')

out1 = join(discoutils, 'gigaw_NPs_in_MR_R2_TechTC_am_maas.txt')
out2 = join(discoutils, 'gigaw_NPs_in_MR_R2_TechTC_am_maas.uniq.100.txt')
out3 = join(discoutils, 'gigaw-obs-wins', 'gigaw-obs-wins.fet')

with temp_chdir(discoutils):
    cmd = 'python {} {} -o {} -s {}'
    run_and_log_output(cmd, script, byblo_features_file, out1, modifiers)

    run_and_log_output("cat {} | sort | uniq -c | awk '$1>{} {print $2}' > {}", out1, thresh, out2)

    mkdirs_if_not_exists(join(discoutils, 'gigaw-obs-wins'))
    run_and_log_output('python discoutils/find_all_NPs.py {} -v -s {} -o {}',
                       byblo_features_file, out2, out3)

with temp_chdir(byblo_base_dir):
    run_and_log_output('./byblo.sh -i {} -o {} -t 10 --stages enumerate,count,filter '
                       '--filter-entry-freq 200 --filter-feature-freq 200 --filter-event-freq 100',
                       out3, dirname(out3))
    run_and_log_output('./unindex-all.sh {}', out3)

with temp_chdir(dirname(out3)):
    obs_vectors_dir = join(prefix, 'FeatureExtractionToolkit', 'observed_vectors')
    mkdirs_if_not_exists(obs_vectors_dir)
    run_and_log_output("awk 'NF>11' gigaw-obs-wins.fet.events.filtered.strings >  {}",
                       join(obs_vectors_dir, 'exp10-13_AN_NNvectors-cleaned'))