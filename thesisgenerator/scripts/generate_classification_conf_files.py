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


class Experiment():
    def __init__(self, number,
                 composer_name, thesaurus_file,
                 labelled_name,
                 unlabelled_name, unlabelled_num,
                 thesaurus_features_name, thesaurus_features_num,
                 document_features, distrib_vector_dim,
                 baronified=False, use_similarity=True,
                 random_neighbour_thesaurus=False,
                 decode_token_handler='SignifiedOnlyFeatureHandler'):
        self.number = number
        self.composer_name = composer_name
        self.thesaurus_file = thesaurus_file
        self.labelled_name = labelled_name
        self.document_features = document_features
        self.baronified = baronified
        self.use_similarity = use_similarity
        self.random_neighbour_thesaurus = random_neighbour_thesaurus
        self.decode_token_handler = decode_token_handler

        # todo this should really be moved to ExpLosion
        if 'socher' in composer_name.lower():
            self.unlabelled_name = '-'
            self.unlabelled_num = '-'
            self.thesaurus_features_name = '-'
            self.thesaurus_features_num = '-'
            self.distrib_vector_dim = 100
        else:
            self.unlabelled_name = unlabelled_name
            self.unlabelled_num = unlabelled_num
            self.thesaurus_features_name = thesaurus_features_name
            self.thesaurus_features_num = thesaurus_features_num
            self.distrib_vector_dim = distrib_vector_dim

    def __str__(self):
        # num: doc_feats, comp, handler, unlab, svd, lab, thes_feats
        return "'%s'" % ','.join([str(self.number),
                                  self.unlabelled_name,
                                  self.labelled_name,
                                  str(self.distrib_vector_dim),
                                  self.composer_name,
                                  self.document_features,
                                  self.thesaurus_features_name,
                                  str(int(self.baronified)),
                                  str(int(self.use_similarity)),
                                  str(int(self.random_neighbour_thesaurus)),
                                  self.decode_token_handler,
                                  ])

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        d1 = deepcopy(self.__dict__)
        d2 = deepcopy(other.__dict__)
        return set(d1.keys()) == set(d2.keys()) and all(d1[x] == d2[x] for x in d1.keys() if x != 'number')


def basic_experiments(exp_number, prefix, composer_algos, use_similarity=True):
    for thesf_num, thesf_name in zip([12, 13], ['dependencies', 'windows']):
        for unlab_num, unlab_name in zip([10], ['gigaw']):  # 11, 'wiki'
            for labelled_corpus in ['R2', 'MR']:
                for svd_dims in [0, 100]:
                    for composer_class in composer_algos:
                        composer_name = composer_class.name

                        if composer_name == 'Baroni' and svd_dims == 0:
                            continue  # not training Baroni without SVD
                        if composer_name == 'APDT' and thesf_name == 'windows':
                            continue  # APDT only works with dependency unigram vectors
                        if composer_name == 'Socher':
                            continue  # Socher RAE done separately below as it only works for a small subset of settings

                        if composer_name == 'Observed':
                            pattern = unred_obs_pattern if svd_dims < 1 else reduced_obs_pattern
                        else:
                            pattern = unred_pattern if svd_dims < 1 else reduced_pattern

                        thesaurus_file = pattern.format(**locals())
                        e = Experiment(exp_number, composer_name, thesaurus_file, labelled_corpus, unlab_name,
                                       unlab_num,
                                       thesf_name, thesf_num, 'AN_NN', svd_dims,
                                       use_similarity=use_similarity)
                        experiments.append(e)
                        exp_number += 1

    # do Socher RAE experiments
    for labelled_corpus in ['R2', 'MR']:
        composer_name = 'Socher'
        e = Experiment(exp_number, composer_name, socher_thesaurus_file, labelled_corpus,
                       'Neuro', 'Neuro', 'Neuro', 'Neuro', 'AN_NN', 0, use_similarity=use_similarity)
        experiments.append(e)
        exp_number += 1
    return exp_number


def an_only_nn_only_experiments(exp_number, prefix, composer_algos):
    # do some experiment with AN or NN features only for comparison
    thesf_num, thesf_name = 12, 'dependencies'  # only dependencies
    unlab_num, unlab_name = 10, 'gigaw'
    svd_dims = 100  # do only 100 dimensional experiments (to include Baroni and Socher)
    for labelled_corpus in ['R2']:
        for doc_feature_type in ['AN', 'NN']:
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

                if e not in experiments:
                    experiments.append(e)
                    exp_number += 1

    return exp_number


def baronified_experiments(exp_number, prefix):
    # do experiments with a reduced thesaurus entry sets, aka "Baronified"
    # todo not using observed-baronified because I haven't built such a thesaurus
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
                                              'socher_vectors/thesaurus_baronified/socher.baronified.sims.neighbours'
                                              '.strings')
            else:
                pattern = '{prefix}/exp{unlab_num}-{thesf_num}bAN_NN_{unlab_name}-{svd_dims}_{composer_name}_baronified/' \
                          'AN_NN_{unlab_name}-{svd_dims}_{composer_name}.baronified.sims.neighbours.strings'
                thesaurus_file = pattern.format(**locals())
            e = Experiment(exp_number, composer_name, thesaurus_file, labelled_corpus, unlab_name, unlab_num,
                           thesf_name, thesf_num, 'AN_NN', svd_dims, baronified=True)
            experiments.append(e)
            exp_number += 1
            print e, ','
    return exp_number


def baselines(exp_number):
    # random-neighbour experiments. These include an "random_neighbour_thesaurus=True" option in the conf file
    for corpus in ['R2', 'MR']:
        e = Experiment(exp_number, 'Random', None, corpus, '-', None, '-', None, 'AN_NN', -1,
                       random_neighbour_thesaurus=True)
        experiments.append(e)
        exp_number += 1

        # signifier experiments (bag-of-words)
        e = Experiment(exp_number, 'Signifier', None, corpus, '-', None, '-', None, 'AN_NN', -1,
                       decode_token_handler='BaseFeatureHandler')
        experiments.append(e)
        exp_number += 1

    return exp_number


prefix = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit'
composer_algos = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer,
                  RightmostWordComposer, BaroniComposer,
                  Bunch(name='Observed'), Bunch(name='APDT'), Bunch(name='Socher')]

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

socher_thesaurus_file = os.path.join(prefix, 'socher_vectors/thesaurus/socher.sims.neighbours.strings')

experiments = []

exp_number = baselines(1)
exp_number = basic_experiments(exp_number, prefix, composer_algos, use_similarity=True)
exp_number = basic_experiments(exp_number, prefix, composer_algos, use_similarity=False)
exp_number = an_only_nn_only_experiments(exp_number, prefix, composer_algos)
# exp_number = baronified_experiments(exp_number, prefix)

for e in experiments:
    print e, ','

# sys.exit(0)
print 'Writing conf files'
megasuperbase_conf_file = 'conf/exp1-superbase.conf'
for exp in experiments:
    # sanity check
    if exp.thesaurus_file and os.path.exists(exp.thesaurus_file):
        print "last modified: %s" % time.ctime(os.path.getmtime(exp.thesaurus_file)), \
            os.stat(exp.thesaurus_file).st_size >> 20, \
            exp.thesaurus_file  # size in MB
    else:
        print 'MISSING THESAURUS:', exp.thesaurus_file

    experiment_dir = 'conf/exp%d' % exp.number
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)

    base_conf_file = os.path.join(experiment_dir, 'exp%d_base.conf' % exp.number)
    shutil.copy(megasuperbase_conf_file, base_conf_file)
    if exp.labelled_name == 'R2':
        train_data = 'sample-data/reuters21578/r8-tagged-grouped'
    else:
        train_data = 'sample-data/movie-reviews-tagged'

    set_in_conf_file(base_conf_file, ['training_data'], train_data)
    set_in_conf_file(base_conf_file, ['feature_extraction', 'decode_token_handler'],
                     'thesisgenerator.plugins.bov_feature_handlers.%s' % exp.decode_token_handler)
    set_in_conf_file(base_conf_file, ['feature_extraction', 'random_neighbour_thesaurus'],
                     exp.random_neighbour_thesaurus)
    set_in_conf_file(base_conf_file, ['vector_sources', 'unigram_paths'],
                     [exp.thesaurus_file] if exp.thesaurus_file else [])
    set_in_conf_file(base_conf_file, ['output_dir'], './conf/exp%d/output' % exp.number)
    set_in_conf_file(base_conf_file, ['name'], 'exp%d' % exp.number)

    requested_features = exp.document_features.split('_')
    for doc_feature_type in ['AN', 'NN', 'VO', 'SVO']:
        set_in_conf_file(base_conf_file, ['feature_extraction', 'extract_%s_features' % doc_feature_type],
                         doc_feature_type in requested_features)

    # do not allow lexical overlap to prevent Left and Right from relying on word identity
    set_in_conf_file(base_conf_file, ['vector_sources', 'allow_lexical_overlap'], False)

    if exp.use_similarity:
        set_in_conf_file(base_conf_file, ['feature_extraction', 'sim_compressor'], 'thesisgenerator.utils.misc.unit')
    else:
        set_in_conf_file(base_conf_file, ['feature_extraction', 'sim_compressor'], 'thesisgenerator.utils.misc.one')