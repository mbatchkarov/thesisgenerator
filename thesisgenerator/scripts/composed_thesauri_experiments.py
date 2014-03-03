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
use this script to generate the conf files required to run them through the classification framework
'''

prefix = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit'
composer_algos = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer,
                  RightmostWordComposer, BaroniComposer, None]  # None stands for observed


class Experiment():
    def __init__(self, number,
                 composer_name, thesaurus_file,
                 labelled_name,
                 unlabelled_name, unlabelled_num,
                 thesaurus_features_name, thesaurus_features_num,
                 document_features, svd):
        self.number = number
        self.composer_name = composer_name
        self.thesaurus_file = thesaurus_file
        self.labelled_name = labelled_name
        self.unlabelled_name = unlabelled_name
        self.unlabelled_num = unlabelled_num
        self.thesaurus_features_name = thesaurus_features_name
        self.thesaurus_features_num = thesaurus_features_num
        self.svd = svd
        self.document_features = document_features

    def __str__(self):
        # num: doc_feats, comp, handler, unlab, svd, lab, thes_feats
        return "'%s'" % ','.join([str(self.number),
                                  self.unlabelled_name,
                                  self.labelled_name,
                                  str(self.svd),
                                  self.composer_name,
                                  self.document_features,
                                  self.thesaurus_features_name])

    def __repr__(self):
        return str(self)


# e.g. exp10-13bAN_NN_gigaw_Left/AN_NN_gigaw_Left.sims.neighbours.strings
unred_pattern = '{prefix}/exp{unlab_num}-{thesf_num}bAN_NN_{unlab_name}_{composer_name}/' \
                'AN_NN_{unlab_name}_{composer_name}.sims.neighbours.strings'
# e.g. exp10-13bAN_NN_gigaw_Observed/exp10.sims.neighbours.strings
unred_obs_pattern = '{prefix}/exp{unlab_num}-{thesf_num}bAN_NN_{unlab_name}_{composer_name}/' \
                    'exp{unlab_num}.sims.neighbours.strings'

# e.g. exp10-12bAN_NN_gigaw-30_Mult/AN_NN_gigaw-30_Mult.sims.neighbours.strings
reduced_pattern = '{prefix}/exp{unlab_num}-{thesf_num}bAN_NN_{unlab_name}-{svd_dims}_{composer_name}/' \
                  'AN_NN_{unlab_name}-{svd_dims}_{composer_name}.sims.neighbours.strings'
# e.g. exp10-12bAN_NN_gigaw-30_Observed/exp10-SVD30.sims.neighbours.strings
reduced_obs_pattern = '{prefix}/exp{unlab_num}-{thesf_num}bAN_NN_{unlab_name}-{svd_dims}_{composer_name}/' \
                      'exp{unlab_num}-SVD{svd_dims}.sims.neighbours.strings'

experiments = []
exp_number = 1
for thesf_num, thesf_name in zip([12, 13], ['dependencies', 'windows']):
    for unlab_num, unlab_name in zip([10], ['gigaw']):  # 11, 'wiki'
        for labelled_corpus in ['R2', 'MR']:
            for svd_dims in [0, 30, 300, 1000]:
                for composer_class in composer_algos:
                    if composer_class == BaroniComposer and svd_dims == 0:
                        continue  # not training Baroni without SVD
                    if composer_class:
                        pattern = unred_pattern if svd_dims < 1 else reduced_pattern
                        composer_name = composer_class.name
                    else:
                        pattern = unred_obs_pattern if svd_dims < 1 else reduced_obs_pattern
                        composer_name = 'Observed'

                    thesaurus_file = pattern.format(**locals())
                    e = Experiment(exp_number, composer_name, thesaurus_file, labelled_corpus, unlab_name, unlab_num,
                                   thesf_name, thesf_num, 'AN_NN', svd_dims)
                    experiments.append(e)
                    exp_number += 1
                    print e, ','

# do a few experiment with AN/ NN features only for comparison
for doc_feature_type in ['AN', 'NN']:
    for composer_class in composer_algos:
        pattern = reduced_pattern
        composer_name = composer_class.name if composer_class else 'Observed'
        thesaurus_file = pattern.format(**locals())
        e = Experiment(exp_number, composer_name, thesaurus_file, 'R2', 'gigaw', 10,
                       'dependencies', 12, doc_feature_type, 300)
        experiments.append(e)
        print e, ','
        exp_number += 1

#  do APDT experiments
thesf_num, thesf_name = 12, 'dependencies' # only dependencies
for unlab_num, unlab_name in zip([10], ['gigaw']):  # 11, , 'wiki'
    for labelled_corpus in ['R2', 'MR']:
        for svd_dims in [0, 30, 300, 1000]:
            pattern = unred_pattern if svd_dims < 1 else reduced_pattern
            composer_name = 'APDT'
            thesaurus_file = pattern.format(**locals())
            e = Experiment(exp_number, composer_name, thesaurus_file, labelled_corpus, unlab_name, unlab_num,
                           thesf_name, thesf_num, 'AN_NN', svd_dims)
            experiments.append(e)
            exp_number += 1
            print e, ','

#  do Socher RAE experiments
for labelled_corpus in ['R2', 'MR']:
    pattern = unred_pattern if svd_dims < 1 else reduced_pattern
    composer_name = 'Socher'
    thesaurus_file = os.path.join(prefix, 'socher_vectors/thesaurus/socher.sims.neighbours.strings')
    e = Experiment(exp_number, composer_name, thesaurus_file, labelled_corpus,
                   'Neuro', 'Neuro', 'Neuro', 'Neuro', 'AN_NN', 0)
    experiments.append(e)
    exp_number += 1
    print e, ','

print 'Writing conf files'
megasuperbase_conf_file = 'conf/exp1-superbase.conf'
for exp in experiments:
    # sanity check
    if not os.path.exists(exp.thesaurus_file):
        print 'MISSING THESAURUS:', exp.thesaurus_file

    experiment_dir = 'conf/exp%d' % exp.number
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)

    base_conf_file = os.path.join(experiment_dir, 'exp%d_base.conf' % exp.number)
    # print base_conf_file, '\t\t', thes
    shutil.copy(megasuperbase_conf_file, base_conf_file)
    if exp.labelled_name == 'R2':
        train_data = 'sample-data/reuters21578/r8train-tagged-grouped'
        test_data = 'sample-data/reuters21578/r8test-tagged-grouped'
    else:
        train_data = 'sample-data/movie-reviews-train-tagged'
        test_data = 'sample-data/movie-reviews-test-tagged'

    set_in_conf_file(base_conf_file, ['vector_sources', 'unigram_paths'], [exp.thesaurus_file])
    set_in_conf_file(base_conf_file, ['output_dir'], './conf/exp%d/output' % exp.number)
    set_in_conf_file(base_conf_file, ['name'], 'exp%d' % exp.number)
    set_in_conf_file(base_conf_file, ['training_data'], train_data)
    set_in_conf_file(base_conf_file, ['test_data'], test_data)
    for doc_feature_type in exp.document_features.split('_'):
        set_in_conf_file(base_conf_file, ['feature_extraction', 'extract_%s_features' % doc_feature_type], True)
    config_obj, configspec_file = parse_config_file(base_conf_file)