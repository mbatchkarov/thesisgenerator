from collections import ChainMap
import os
import sys

sys.path.append('.')
from datetime import datetime as dt
from discoutils.misc import Bunch
from thesisgenerator.utils import db
from thesisgenerator.composers.vectorstore import *
import numpy as np

# populate database

def get_size(thesaurus_file):
    if os.path.exists(thesaurus_file):
        modified = dt.fromtimestamp(os.path.getmtime(thesaurus_file))
        size = os.stat(thesaurus_file).st_size >> 20  # size in MB
    else:
        modified, size = None, None
    gz_file = thesaurus_file + '.gz'
    gz_size = os.stat(gz_file).st_size >> 20 if os.path.exists(gz_file) else None
    return modified, size, gz_size


if __name__ == '__main__':
    prefix = '/mnt/lustre/scratch/inf/mmb28/FeatureExtractionToolkit'
    composer_algos = [AdditiveComposer, MultiplicativeComposer,
                      LeftmostWordComposer, RightmostWordComposer,
                      BaroniComposer, Bunch(name='Observed')]

    # e.g. exp10-12-composed-ngrams/AN_NN_gigaw_Add.events.filtered.strings
    unred_pattern = '{prefix}/exp{unlab_num}-{thesf_num}-composed-ngrams/' \
                    'AN_NN_{unlab_name:.5}_{composer_name}.events.filtered.strings'

    # e.g. exp10-12-composed-ngrams/AN_NN_gigaw-100_Add.events.filtered.strings
    reduced_pattern = '{prefix}/exp{unlab_num}-{thesf_num}-composed-ngrams/' \
                      'AN_NN_{unlab_name:.5}-{svd_dims}_{composer_name}.events.filtered.strings'

    # random neighbours composer for use in baselines
    v = db.Vectors.create(algorithm='random_neigh',
                          dimensionality=None, unlabelled_percentage=None,
                          unlabelled=None, composer='random_neigh')


    # random VECTORS for use in baseline
    path = '/mnt/lustre/scratch/inf/mmb28/FeatureExtractionToolkit/random_vectors.gz'
    modified, size, gz_size = get_size(path)
    v = db.Vectors.create(algorithm='random_vect', composer='random_vect', path=path,
                          dimensionality=None, unlabelled_percentage=None,
                          modified=modified, size=size, gz_size=gz_size)

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

                    pattern = unred_pattern if svd_dims < 1 else reduced_pattern
                    thesaurus_file = pattern.format(**locals())
                    modified, size, gz_size = get_size(thesaurus_file)
                    v = db.Vectors.create(algorithm='count_' + thesf_name,
                                          dimensionality=svd_dims, unlabelled=unlab_name,
                                          path=thesaurus_file, composer=composer_name,
                                          modified=modified, size=size, gz_size=gz_size)
                    print(v)

    # Socher (2011)'s paraphrase model, and the same with simple composition
    for composer_class in [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer,
                           RightmostWordComposer, Bunch(name='Socher')]:
        composer_name = composer_class.name
        pattern = '{prefix}/socher_vectors/composed/AN_NN_turian_{composer_name}.events.filtered.strings'
        thesaurus_file = pattern.format(**locals())
        modified, size, gz_size = get_size(thesaurus_file)
        v = db.Vectors.create(algorithm='turian',
                              dimensionality=100,
                              unlabelled='turian',
                              path=thesaurus_file, composer=composer_name,
                              modified=modified, size=size, gz_size=gz_size)
        print(v)

    # GloVe vectors with simple composition
    for composer_class in [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer, RightmostWordComposer]:
        composer_name = composer_class.name
        pattern = '{prefix}/glove/AN_NN_glove-gigaw_{composer_name}.events.filtered.strings'
        thesaurus_file = pattern.format(**locals())
        modified, size, gz_size = get_size(thesaurus_file)
        v = db.Vectors.create(algorithm='glove',
                              dimensionality=100, unlabelled='gigaw',
                              path=thesaurus_file, composer=composer_name,
                              modified=modified, size=size, gz_size=gz_size)
        print(v)

    # word2vec composed with various simple algorithms, including varying amounts of unlabelled data
    pattern2 = '{prefix}/word2vec_vectors/composed/AN_NN_word2vec_{percent}percent-rep{rep}_' \
               '{composer}.events.filtered.strings'
    # some files were created with this format (float percent representation instead of int)
    pattern1 = '{prefix}/word2vec_vectors/composed/AN_NN_word2vec_{percent:.2f}percent-rep{rep}_' \
               '{composer}.events.filtered.strings'
    # and some files were done on Wikipedia and have a different naming scheme
    pattern3 = '{prefix}/word2vec_vectors/composed/AN_NN_word2vec-wiki_{percent:.2f}percent-rep{rep}_' \
               '{composer}.events.filtered.strings'


    def _do_w2v_vectors(unlabelled='gigaw'):
        global prefix
        for composer_class in [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer, RightmostWordComposer]:
            composer = composer_class.name
            for rep in range(1 if percent < 100 else 3):
                thesaurus_file = pattern1.format(**ChainMap(locals(), globals()))
                if not os.path.exists(thesaurus_file):
                    thesaurus_file = pattern2.format(**ChainMap(locals(), globals()))
                if unlabelled == 'wiki':
                    thesaurus_file = pattern3.format(**ChainMap(locals(), globals()))

                modified, size, gz_size = get_size(thesaurus_file)
                v = db.Vectors.create(algorithm='word2vec', dimensionality=100,
                                      unlabelled=unlabelled, path=thesaurus_file, unlabelled_percentage=percent,
                                      composer=composer, modified=modified, size=size, gz_size=gz_size,
                                      rep=rep)
                print(v)


    for percent in range(100, 9, -10):
        _do_w2v_vectors()

    for percent in range(1, 10):
        _do_w2v_vectors()

    for percent in np.arange(0.01, 0.92, .1):
        _do_w2v_vectors()

    for percent in [15, 50]:
        _do_w2v_vectors('wiki')

    # count vectors with PPMI, no SVD
    ppmi_file_pattern = '{prefix}/exp{unlab_num}-{thesf_num}-composed-ngrams-ppmi/' \
                        'AN_NN_{unlab_name:.5}_{composer_name}.events.filtered.strings'
    for composer_class in [AdditiveComposer, MultiplicativeComposer,
                           LeftmostWordComposer, RightmostWordComposer]:
        for thesf_num, thesf_name in zip([12, 13], ['dependencies', 'windows']):
            for unlab_num, unlab_name in zip([10], ['gigaw']):
                composer_name = composer_class.name

                thesaurus_file = ppmi_file_pattern.format(**locals())
                modified, size, gz_size = get_size(thesaurus_file)
                v = db.Vectors.create(algorithm='count_' + thesf_name, use_ppmi=True,
                                      dimensionality=0, unlabelled=unlab_name,
                                      path=thesaurus_file, composer=composer_name,
                                      modified=modified, size=size, gz_size=gz_size)
                print(v)