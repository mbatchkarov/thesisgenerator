from copy import deepcopy
import os
import shutil
import sys
import time
from discoutils.misc import Bunch

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
                  RightmostWordComposer, BaroniComposer, Bunch(name='Observed')]


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

    def __eq__(self, other):
        d1 = deepcopy(self.__dict__)
        d2 = deepcopy(other.__dict__)
        return set(d1.keys()) == set(d2.keys()) and all(d1[x] == d2[x] for x in d1.keys() if x != 'number')

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
            for svd_dims in [0, 100]:
                for composer_class in composer_algos:
                    composer_name = composer_class.name
                    if composer_name == 'Baroni' and svd_dims == 0:
                        continue  # not training Baroni without SVD
                    elif composer_name == 'Observed':
                        pattern = unred_obs_pattern if svd_dims < 1 else reduced_obs_pattern
                    else:
                        pattern = unred_pattern if svd_dims < 1 else reduced_pattern

                    thesaurus_file = pattern.format(**locals())
                    e = Experiment(exp_number, composer_name, thesaurus_file, labelled_corpus, unlab_name, unlab_num,
                                   thesf_name, thesf_num, 'AN_NN', svd_dims)
                    experiments.append(e)
                    exp_number += 1
                    print e, ','

# do a few experiment with AN/ NN features only for comparison
thesf_num, thesf_name = 12, 'dependencies'  # only dependencies
unlab_num, unlab_name = 10, 'gigaw'
svd_dims = 100
labelled_corpus = 'R2'
for doc_feature_type in ['AN', 'NN']:
    for composer_class in composer_algos:
        pattern = reduced_pattern
        composer_name = composer_class.name
        if composer_name == 'Observed':
            pattern = reduced_obs_pattern
        else:
            pattern = reduced_pattern

        thesaurus_file = pattern.format(**locals())
        e = Experiment(exp_number, composer_name, thesaurus_file, labelled_corpus, unlab_name, unlab_num,
                       thesf_name, thesf_num, doc_feature_type, svd_dims)
        experiments.append(e)
        print e, ','
        exp_number += 1

#  do APDT experiments
thesf_num, thesf_name = 12, 'dependencies'  # only dependencies
composer_name = 'APDT'
for unlab_num, unlab_name in zip([10], ['gigaw']):  # 11, , 'wiki'
    for labelled_corpus in ['R2', 'MR']:
        for svd_dims in [0, 100]:
            pattern = unred_pattern if svd_dims < 1 else reduced_pattern
            thesaurus_file = pattern.format(**locals())
            e = Experiment(exp_number, composer_name, thesaurus_file, labelled_corpus, unlab_name, unlab_num,
                           thesf_name, thesf_num, 'AN_NN', svd_dims)
            experiments.append(e)
            exp_number += 1
            print e, ','

#  do Socher RAE experiments
socher_thesaurus_file = os.path.join(prefix, 'socher_vectors/thesaurus/socher.sims.neighbours.strings')
for labelled_corpus in ['R2', 'MR']:
    composer_name = 'Socher'
    e = Experiment(exp_number, composer_name, socher_thesaurus_file, labelled_corpus,
                   'Neuro', 'Neuro', 'Neuro', 'Neuro', 'AN_NN', 0)
    experiments.append(e)
    exp_number += 1
    print e, ','

# do RAE/APDT with AN-only or NN-only; AN-only or NN-only on MR corpus
# these experiments were omitted above to save time
composer_algos.append(Bunch(name='APDT'))
composer_algos.append(Bunch(name='Socher'))
unlab_num, unlab_name = 10, 'gigaw'
svd_dims = 100
for doc_feature_type in ['AN', 'NN']:
    for labelled_corpus in ['R2', 'MR']:
        for composer_class in composer_algos:
            composer_name = composer_class.name

            if composer_name == 'Socher':
                thesaurus_file = socher_thesaurus_file
            else:
                if composer_name == 'Observed':
                    pattern = reduced_obs_pattern
                else:
                    pattern = reduced_pattern
                thesaurus_file = pattern.format(**locals())

            e = Experiment(exp_number, composer_name, thesaurus_file, labelled_corpus, unlab_name, unlab_num,
                           thesf_name, thesf_num, doc_feature_type, svd_dims)
            if e not in experiments:  # this may generate experiments that have been done before, ignore them
                experiments.append(e)
                print e, ','
                exp_number += 1

print '-----------'
# do experiments with a reduced thesaurus entry sets, aka "Baronified"
composer_algos = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer,
                  RightmostWordComposer, BaroniComposer, Bunch(name='APDT'), Bunch(name='Socher')]
svd_dims = 100
thesf_num, thesf_name = 12, 'dependencies'  # only dependencies
unlab_num, unlab_name = 10, 'gigaw'
for labelled_corpus in ['R2', 'MR']:
    for composer_class in composer_algos:
        composer_name = composer_class.name
        if composer_name == 'Socher':
            thesaurus_file = os.path.join(prefix,
                                          'socher_vectors/thesaurus_baronified/socher.baronified.sims.neighbours.strings')
        else:
            pattern = '{prefix}/exp{unlab_num}-{thesf_num}bAN_NN_{unlab_name}-{svd_dims}_{composer_name}_baronified/' \
                      'AN_NN_{unlab_name}-{svd_dims}_{composer_name}.baronified.sims.neighbours.strings'
            thesaurus_file = pattern.format(**locals())
        e = Experiment(exp_number, composer_name, thesaurus_file, labelled_corpus, unlab_name, unlab_num,
                       thesf_name, thesf_num, 'AN_NN', svd_dims)
        experiments.append(e)
        exp_number += 1
        print e, ','

exp_number += 2 # set aside 2 numbers for random-neighbour experiments. These are based on 87 and 96
#  (because these are the smallest real thesauri and I have to load them due to silly code) and
# include random_neighbour_thesaurus=True

print 'Writing conf files'
megasuperbase_conf_file = 'conf/exp1-superbase.conf'
for exp in experiments:
    # sanity check
    if os.path.exists(exp.thesaurus_file):
        print "last modified: %s" % time.ctime(os.path.getmtime(exp.thesaurus_file)), exp.thesaurus_file
    else:
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

    requested_features = exp.document_features.split('_')
    for doc_feature_type in ['AN', 'NN', 'VO', 'SVO']:
        set_in_conf_file(base_conf_file, ['feature_extraction', 'extract_%s_features' % doc_feature_type],
                         doc_feature_type in requested_features)
    config_obj, configspec_file = parse_config_file(base_conf_file)