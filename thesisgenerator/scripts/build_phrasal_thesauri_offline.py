import logging
import os
from thesisgenerator.utils.cmd_utils import set_stage_in_byblo_conf_file, run_byblo, parse_byblo_conf_file, reindex_all_byblo_vectors, run_and_log_output

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s\t%(module)s.%(funcName)s ""(line %(lineno)d)\t%(levelname)s : %(""message)s")

os.chdir('/Volumes/LocalDataHD/mmb28/NetBeansProjects/Byblo-2.2.0') # byblo path
byblo_conf_file = '/Volumes/LocalDataHD/mmb28/NetBeansProjects/Byblo-2.2.0/sample-data/conf.txt'

opts, _ = parse_byblo_conf_file(byblo_conf_file)

## calculate vectors for all entries
#set_stage_in_byblo_conf_file(byblo_conf_file, 1)
#run_byblo(byblo_conf_file)
#set_stage_in_byblo_conf_file(byblo_conf_file, 0)
## get vectors as strings
#unindex_all_byblo_vectors(opts.input)

# mess with vectors, add to/modify entries and events files
# whether to modify the features file is less obvious- do composed entries have different features to the non-composed ones?
# assume we have a file with all composed vectors, one per dataset, per composer
composed_entries = '/Volumes/LocalDataHD/mmb28/NetBeansProjects/thesisgenerator/bigram_wiki_Mult.entries.txt'
composed_vectors = '/Volumes/LocalDataHD/mmb28/NetBeansProjects/thesisgenerator/bigram_wiki_Mult.vectors.tsv'
#composed_features = '/Volumes/LocalDataHD/mmb28/NetBeansProjects/thesisgenerator/bigram_wiki_Mult.features.txt'

# todo this appending should only occur once, should I check for it?
run_and_log_output('cat {} >> {}.entries.filtered.strings'.format(composed_entries, opts.input))
run_and_log_output('cat {} >> {}.events.filtered.strings'.format(composed_vectors, opts.input))
# todo features of newly created entries must be the same as these of the old ones, or the features file will have to be updated too
#run_and_log_output('cat {} > {}.features.filtered.strings'.format(composed_features, opts.input))

# restore indices from strings
reindex_all_byblo_vectors(opts.input)

# re-run all-pairs similarity
set_stage_in_byblo_conf_file(byblo_conf_file, 2)

run_byblo(byblo_conf_file)
set_stage_in_byblo_conf_file(byblo_conf_file, 0)

