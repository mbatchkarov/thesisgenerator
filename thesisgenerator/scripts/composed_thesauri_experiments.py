import os
import shutil
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
from thesisgenerator.composers.vectorstore import *
from thesisgenerator.utils.conf_file_utils import set_in_conf_file, parse_config_file

'''
Once all thesauri with ngram entries (obtained using different composition methods) have been built offline,
use this script to run them through the classification framework
'''

__author__ = 'mmb28'

handler = 'thesisgenerator.plugins.bov_feature_handlers.SignifiedFeatureHandler'
prefix = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit'
composer_algos = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer,
                  RightmostWordComposer, MinComposer, MaxComposer]

# get the names of all GIGAW unreduced thesauri
pattern = '{1}/exp10-13bAN_NN_gigaw_{0}/AN_NN_gigaw_{0}.sims.neighbours.strings'
gigaw_unreduced_thesauri = [pattern.format(c.name, prefix) for c in composer_algos]
# the names of the unreduced observed thesauri aren't consistent with all other names, handle separately
gigaw_unreduced_thesauri.append('{1}/exp10-13bAN_NN_gigaw_{0}/exp10.sims.neighbours.strings'.format('Observed', prefix))

# get the names of all WIKI unreduced thesauri
pattern = '{1}/exp11-13bAN_NN_wiki_{0}/AN_NN_wiki_{0}.sims.neighbours.strings'
wiki_unreduced_thesauri = [pattern.format(c.name, prefix) for c in composer_algos]
wiki_unreduced_thesauri.append('{1}/exp11-13bAN_NN_wiki_{0}/exp11.sims.neighbours.strings'.format('Observed', prefix))

thesauri = {(0, 42): gigaw_unreduced_thesauri, (0, 56): gigaw_unreduced_thesauri,
            (0, 49): wiki_unreduced_thesauri, (0, 63): wiki_unreduced_thesauri,
            (0, 70): gigaw_unreduced_thesauri, (0, 77): gigaw_unreduced_thesauri}

# INSERT SVD-REDUCED THESAURI
composer_algos.append(BaroniComposer)
superbase = 84
for labelled_corpus in ['R2', 'MR']:
    for svd_dims in [30, 300, 1000]:
        for number, name in zip([10, 11], ['gigaw', 'wiki']):
            print '{}\t== {}- {}, {}'.format(superbase, labelled_corpus, svd_dims, name)
            pattern = '{0}/exp{3}-13bAN_NN_{4}-{2}_{1}/AN_NN_{4}-{2}_{1}.sims.neighbours.strings'
            obs_pattern = '{0}/exp{3}-13bAN_NN_{4}-{2}_{1}/exp{3}-SVD{2}.sims.neighbours.strings'
            files = []
            files = [pattern.format(prefix, c.name, svd_dims, number, name) for c in composer_algos]
            files.append(obs_pattern.format(prefix, 'Observed', svd_dims, number, name))
            thesauri[(svd_dims, superbase)] = files
            superbase += len(files)

for k in thesauri.keys():
    svd_dims, first_exp = k
    superbase_conf_file = 'conf/exp%d-superbase.conf' % first_exp

    for offset, thes in enumerate(thesauri[k]):
        # sanity check
        assert os.path.exists(thes)

        experiment_dir = 'conf/exp%d' % (first_exp + offset)
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)

        base_conf_file = os.path.join(experiment_dir, 'exp%d_base.conf' % (first_exp + offset))
        # print base_conf_file, '\t\t', thes
        shutil.copy(superbase_conf_file, base_conf_file)

        set_in_conf_file(base_conf_file, ['vector_sources', 'unigram_paths'], [thes])
        set_in_conf_file(base_conf_file, ['output_dir'], './conf/exp%d/output' % (first_exp + offset))
        set_in_conf_file(base_conf_file, ['name'], 'exp%d' % (first_exp + offset))
        config_obj, configspec_file = parse_config_file(base_conf_file)

        # print a brief description of the experiment setting for use in plotting
        composer = thes.split('/')[-2].split('_')[-1]
        corpus = 'wiki' if 'wiki' in thes else 'gigaw'
        labelled_corpus = 'MR' if 'movie' in config_obj['training_data'] else 'R2'
        print "{}: 'AN_NN, {}, signified, {}-{}, {}'".format(first_exp + offset, composer,
                                                             corpus, svd_dims, labelled_corpus)
