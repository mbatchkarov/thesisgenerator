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

superbase_conf_file = 'conf/exp35-superbase.conf'
composer_algos = [AdditiveComposer, MultiplicativeComposer, HeadWordComposer,
                  TailWordComposer, MinComposer, MaxComposer]
thesauri = [
    '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp10-12AN_NN_gigaw_{}' \
    '/exp10.sims.neighbours.strings'.format(c.name) for c in composer_algos
]
thesauri.append('Observed') # todo path to thesaurus with observed AN/NN vectors added in

handler = 'thesisgenerator.plugins.bov_feature_handlers.SignifiedFeatureHandler'


# for each Thesaurus
first_exp = 35
for offset, thes in enumerate(thesauri):
    experiment_dir = 'conf/exp%d' % (first_exp + offset)
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    else:
        # todo do not remove this dir, may contain useful results
        shutil.rmtree(experiment_dir)
        os.mkdir(experiment_dir)
    base_conf_file = os.path.join(experiment_dir, 'exp%d_base.conf' % (first_exp + offset))
    shutil.copy(superbase_conf_file, base_conf_file)

    set_in_conf_file(base_conf_file, ['vector_sources', 'unigram_paths'], [thes])
    set_in_conf_file(base_conf_file, ['output_dir'], './conf/exp%d/output' % (first_exp + offset))
    config_obj, configspec_file = parse_config_file(base_conf_file)
    print base_conf_file
    print config_obj['vector_sources']['unigram_paths']

    #run_experiment(first_exp + offset)
    run_and_log_output('qsub -N composed{0} go-single-experiment.sh {0}'.format(first_exp + offset))
