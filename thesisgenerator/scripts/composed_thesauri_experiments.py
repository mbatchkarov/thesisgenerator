import os
import shutil
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
from thesisgenerator.utils.cmd_utils import run_and_log_output
from thesisgenerator.composers.vectorstore import *
from thesisgenerator.plugins.experimental_utils import run_experiment
from thesisgenerator.utils.conf_file_utils import set_in_conf_file, parse_config_file

'''
Once all thesauri with ngram entries (obtained using different composition methods) have been built offline,
use this script to run them through the classification framework
'''

__author__ = 'mmb28'

superbase_conf_file = 'conf/exp42-superbase.conf'
composer_algos = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer,
                  RightmostWordComposer, MinComposer, MaxComposer]

# get the names of all GIGAW thesauri
pattern = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp10-12bAN_NN_gigaw_{0}' \
          '/AN_NN_gigaw_{0}.sims.neighbours.strings'
gigaw_thesauri = [pattern.format(c.name) for c in composer_algos]
gigaw_thesauri.append(pattern.format('Observed'))

# get the names of all WIKI thesauri
pattern = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp11-12bAN_NN_wiki_{0}' \
          '/AN_NN_wiki_{0}.sims.neighbours.strings'
wiki_thesauri = [pattern.format(c.name) for c in composer_algos]
wiki_thesauri.append(pattern.format('Observed'))

handler = 'thesisgenerator.plugins.bov_feature_handlers.SignifiedFeatureHandler'

thesauri = {42: gigaw_thesauri, 56: gigaw_thesauri, 49: wiki_thesauri, 63: wiki_thesauri,
            70: gigaw_thesauri, 77: gigaw_thesauri}

for first_exp in [42, 49, 56, 63, 70, 77]:
    for offset, thes in enumerate(thesauri[first_exp]):

        # skip Observed for now, thesauri aren't ready
        if 'Observed' in thes:
            continue

        experiment_dir = 'conf/exp%d' % (first_exp + offset)
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)
        else:
            # todo do not remove this dir, may contain useful results
            shutil.rmtree(experiment_dir)
            os.mkdir(experiment_dir)
        base_conf_file = os.path.join(experiment_dir, 'exp%d_base.conf' % (first_exp + offset))
        print base_conf_file, thes
        shutil.copy(superbase_conf_file, base_conf_file)

        set_in_conf_file(base_conf_file, ['vector_sources', 'unigram_paths'], [thes])
        set_in_conf_file(base_conf_file, ['output_dir'], './conf/exp%d/output' % (first_exp + offset))
        set_in_conf_file(base_conf_file, ['name'], 'exp%d' % (first_exp + offset))
        config_obj, configspec_file = parse_config_file(base_conf_file)

        #run_experiment(first_exp + offset)
        run_and_log_output('qsub -N composed{0} go-single-experiment.sh {0}'.format(first_exp + offset))
