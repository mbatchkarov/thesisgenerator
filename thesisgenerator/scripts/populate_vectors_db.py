import os
import sys

sys.path.append('.')
from datetime import datetime as dt
from discoutils.misc import Bunch
from thesisgenerator.utils import db
from thesisgenerator.composers.vectorstore import *


# populate database


prefix = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit'
composer_algos = [AdditiveComposer, MultiplicativeComposer,
                  LeftmostWordComposer, RightmostWordComposer,
                  BaroniComposer, Bunch(name='Observed')]

# e.g. exp10-13bAN_NN_gigaw_Left/AN_NN_gigaw_Left.events.filtered.strings
unred_pattern = '{prefix}/exp{unlab_num}-{thesf_num}bAN_NN_{unlab_name:.5}_{composer_name}/' \
                'AN_NN_{unlab_name:.5}_{composer_name}.events.filtered.strings'

# e.g. exp10-13bAN_NN_gigaw_Observed/exp10.events.filtered.strings
unred_obs_pattern = '{prefix}/exp{unlab_num}-{thesf_num}bAN_NN_{unlab_name:.5}_{composer_name}/' \
                    'exp{unlab_num}.events.filtered.strings'

# e.g. exp10-12bAN_NN_gigaw-30_Mult/AN_NN_gigaw-30_Mult.events.filtered.strings
reduced_pattern = '{prefix}/exp{unlab_num}-{thesf_num}bAN_NN_{unlab_name:.5}-{svd_dims}_{composer_name}/' \
                  'AN_NN_{unlab_name:.5}-{svd_dims}_{composer_name}.events.filtered.strings'
# e.g. exp10-12bAN_NN_gigaw-30_Observed/exp10-SVD30.events.filtered.strings
reduced_obs_pattern = '{prefix}/exp{unlab_num}-{thesf_num}bAN_NN_{unlab_name:.5}-{svd_dims}_{composer_name}/' \
                      'exp{unlab_num}-SVD{svd_dims}.events.filtered.strings'

# random neighbours composer for use in baselines
v = db.Vectors.create(algorithm='random',
                      dimensionality=None, unlabelled_percentage=None,
                      unlabelled=None, composer='random')


def get_size(thesaurus_file):
    global modified, size
    if os.path.exists(thesaurus_file):
        modified = dt.fromtimestamp(os.path.getmtime(thesaurus_file))
        size = os.stat(thesaurus_file).st_size >> 20  # size in MB
    else:
        modified, size = None, None
    gz_file = thesaurus_file + '.gz'
    gz_size = os.stat(gz_file).st_size >> 20 if os.path.exists(gz_file) else None
    return modified, size, gz_size

# standard windows/dependency thesauri that I built back in the day
for thesf_num, thesf_name in zip([12, 13], ['dependencies', 'windows']):
    for unlab_num, unlab_name in zip([10], ['gigaw']):
        for svd_dims in [0, 100]:
            for composer_class in composer_algos:
                composer_name = composer_class.name

                if composer_name == 'Baroni' and svd_dims == 0:
                    continue  # not training Baroni without SVD
                if thesf_name == 'dependencies' and composer_name in ['Baroni', 'Observed']:
                    continue  # can't easily run Julie's observed vectors code, so pretend it doesnt exist
                if composer_name == 'Observed':
                    pattern = unred_obs_pattern if svd_dims < 1 else reduced_obs_pattern
                else:
                    pattern = unred_pattern if svd_dims < 1 else reduced_pattern

                thesaurus_file = pattern.format(**locals())
                modified, size, gz_size = get_size(thesaurus_file)
                v = db.Vectors.create(algorithm='count_' + thesf_name, can_build=True,
                                      dimensionality=svd_dims, unlabelled=unlab_name,
                                      path=thesaurus_file, composer=composer_name,
                                      modified=modified, size=size, gz_size=gz_size)
                print(v)

# Socher (2011)'s paraphrase model
path = os.path.join(prefix, 'socher_vectors/thesaurus/socher.events.filtered.strings')
modified, size, gz_size = get_size(path)
db.Vectors.create(algorithm='turian', an_build=False, dimensionality=100,
                  unlabelled='turian', path=path, composer='Socher',
                  modified=modified, size=size, gz_size=gz_size)

# Socher's vectors with simple composition
for composer_class in [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer, RightmostWordComposer]:
    composer_name = composer_class.name
    pattern = '{prefix}/exp12-14bAN_NN_neuro_{composer_name}/AN_NN_neuro_{composer_name}.events.filtered.strings'
    thesaurus_file = pattern.format(**locals())
    modified, size, gz_size = get_size(thesaurus_file)
    v = db.Vectors.create(algorithm='turian',
                          dimensionality=100, unlabelled='turian',
                          path=thesaurus_file, composer=composer_name,
                          modified=modified, size=size, gz_size=gz_size)
    print(v)

# word2vec composed with various simple algorithms, including varying amounts of unlabelled data
pattern = '{prefix}/word2vec_vectors/composed/AN_NN_word2vec_{percent}percent-rep{rep}_' \
          '{composer}.events.filtered.strings'
for percent in range(100, 9, -10):
    for composer_class in [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer, RightmostWordComposer]:
        composer = composer_class.name
        modified, size, gz_size = get_size(thesaurus_file)
        for rep in range(1 if percent < 100 else 3):
            thesaurus_file = pattern.format(**locals())
            v = db.Vectors.create(algorithm='word2vec', dimensionality=100,
                                  unlabelled='gigaw', path=thesaurus_file, unlabelled_percentage=percent,
                                  composer=composer, modified=modified, size=size, gz_size=gz_size,
                                  rep=rep)
            print(v)