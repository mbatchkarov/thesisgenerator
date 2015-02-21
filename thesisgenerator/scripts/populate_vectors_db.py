"""populate database with all available vectors"""
from collections import ChainMap
import os
import sys

sys.path.append('.')
from datetime import datetime as dt
from discoutils.misc import Bunch
from thesisgenerator.utils import db
from thesisgenerator.composers.vectorstore import (AdditiveComposer, MultiplicativeComposer,
                                                   LeftmostWordComposer, RightmostWordComposer, BaroniComposer)


def _get_size(thesaurus_file):
    if os.path.exists(thesaurus_file):
        modified = dt.fromtimestamp(os.path.getmtime(thesaurus_file))
        size = os.stat(thesaurus_file).st_size >> 20  # size in MB
    else:
        modified, size = None, None
    gz_file = thesaurus_file + '.gz'
    gz_size = os.stat(gz_file).st_size >> 20 if os.path.exists(gz_file) else None
    return modified, size, gz_size


def _do_w2v_vectors(percent, unlabelled='gigaw'):
    """word2vec composed with various simple algorithms, including varying amounts of unlabelled data"""
    gigaw_rep_pattern = '{prefix}/word2vec_vectors/composed/' \
                        'AN_NN_word2vec-{unlabelled}_{percent}percent-rep{rep}_{composer}.events.filtered.strings'
    gigaw_avg_pattern = '{prefix}/word2vec_vectors/composed/' \
                        'AN_NN_word2vec-gigaw_100percent-avg3_{composer}.events.filtered.strings'
    # and some files were done on Wikipedia and have a different naming scheme
    wiki_pattern = '{prefix}/word2vec_vectors/composed/' \
                   'AN_NN_word2vec-wiki_{percent:d}percent-rep0_{composer}.events.filtered.strings'

    for composer_class in [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer, RightmostWordComposer]:
        composer = composer_class.name
        for rep in range(1 if percent < 100 else 3):
            thesaurus_file = gigaw_rep_pattern.format(**ChainMap(locals(), globals()))
            if unlabelled == 'wiki':
                thesaurus_file = wiki_pattern.format(**ChainMap(locals(), globals()))

            modified, size, gz_size = _get_size(thesaurus_file)
            v = db.Vectors.create(algorithm='word2vec', dimensionality=100,
                                  unlabelled=unlabelled, path=thesaurus_file, unlabelled_percentage=percent,
                                  composer=composer, modified=modified, size=size, gz_size=gz_size,
                                  rep=rep)
            print(v)
        if percent == 100 and unlabelled == 'gigaw':
            # also include average of the three runs at 100% of all data
            thesaurus_file = gigaw_avg_pattern.format(**ChainMap(locals(), globals()))
            modified, size, gz_size = _get_size(thesaurus_file)
            v = db.Vectors.create(algorithm='word2vec', dimensionality=100,
                                  unlabelled=unlabelled, path=thesaurus_file, unlabelled_percentage=percent,
                                  composer=composer, modified=modified, size=size, gz_size=gz_size,
                                  rep=-1)  # -1 signifies averaging across multiple runs


def _ppmi_vectors(unlab_nums, unlab_names):
    """ count vectors with PPMI, no SVD"""
    ppmi_file_pattern = '{prefix}/exp{unlab_num}-{thesf_num}-composed-ngrams-ppmi/' \
                        'AN_NN_{unlab_name:.5}_{composer_name}.events.filtered.strings'
    for composer_class in [AdditiveComposer, MultiplicativeComposer,
                           LeftmostWordComposer, RightmostWordComposer]:
        for thesf_num, thesf_name in zip([12, 13], ['dependencies', 'windows']):
            for unlab_num, unlab_name in zip(unlab_nums, unlab_names):
                composer_name = composer_class.name

                thesaurus_file = ppmi_file_pattern.format(**ChainMap(locals(), globals()))
                modified, size, gz_size = _get_size(thesaurus_file)
                v = db.Vectors.create(algorithm='count_' + thesf_name, use_ppmi=True,
                                      dimensionality=0, unlabelled=unlab_name,
                                      path=thesaurus_file, composer=composer_name,
                                      modified=modified, size=size, gz_size=gz_size)
                print(v)


def _glove_vectors_wiki():
    # GloVe vectors with simple composition
    for composer_class in [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer, RightmostWordComposer]:
        composer_name = composer_class.name
        pattern = '{prefix}/glove/AN_NN_glove-wiki_{composer_name}.events.filtered.strings'
        thesaurus_file = pattern.format(**ChainMap(locals(), globals()))
        modified, size, gz_size = _get_size(thesaurus_file)
        v = db.Vectors.create(algorithm='glove',
                              dimensionality=100, unlabelled='wiki',
                              path=thesaurus_file, composer=composer_name,
                              modified=modified, size=size, gz_size=gz_size)
        print(v)


def _count_vectors_gigaw():
    """ standard windows/dependency thesauri that I built back in the day"""
    composer_algos = [AdditiveComposer, MultiplicativeComposer,
                      LeftmostWordComposer, RightmostWordComposer,
                      BaroniComposer, Bunch(name='Observed')]

    # e.g. exp10-12-composed-ngrams/AN_NN_gigaw_Add.events.filtered.strings
    unred_pattern = '{prefix}/exp{unlab_num}-{thesf_num}-composed-ngrams/' \
                    'AN_NN_{unlab_name:.5}_{composer_name}.events.filtered.strings'

    # e.g. exp10-12-composed-ngrams/AN_NN_gigaw-100_Add.events.filtered.strings
    reduced_pattern = '{prefix}/exp{unlab_num}-{thesf_num}-composed-ngrams/' \
                      'AN_NN_{unlab_name:.5}-{svd_dims}_{composer_name}.events.filtered.strings'

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
                    thesaurus_file = pattern.format(**ChainMap(locals(), globals()))
                    modified, size, gz_size = _get_size(thesaurus_file)
                    v = db.Vectors.create(algorithm='count_' + thesf_name,
                                          dimensionality=svd_dims, unlabelled=unlab_name,
                                          path=thesaurus_file, composer=composer_name,
                                          modified=modified, size=size, gz_size=gz_size)
                    print(v)


def _turian_vectors():
    # Socher (2011)'s paraphrase model, and the same with simple composition
    for composer_class in [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer,
                           RightmostWordComposer, Bunch(name='Socher')]:
        composer_name = composer_class.name
        pattern = '{prefix}/socher_vectors/composed/AN_NN_turian_{composer_name}.events.filtered.strings'
        thesaurus_file = pattern.format(**ChainMap(locals(), globals()))
        modified, size, gz_size = _get_size(thesaurus_file)
        v = db.Vectors.create(algorithm='turian',
                              dimensionality=100,
                              unlabelled='turian',
                              path=thesaurus_file, composer=composer_name,
                              modified=modified, size=size, gz_size=gz_size)
        print(v)


def _random_baselines():
    """random neighbours/vectors composer baselines"""
    db.Vectors.create(algorithm='random_neigh',
                          dimensionality=None, unlabelled_percentage=None,
                          unlabelled=None, composer='random_neigh')
    # random VECTORS for use in baseline
    path = '/mnt/lustre/scratch/inf/mmb28/FeatureExtractionToolkit/random_vectors.gz'
    modified, size, gz_size = _get_size(path)
    db.Vectors.create(algorithm='random_vect', composer='random_vect', path=path,
                          dimensionality=None, unlabelled_percentage=None,
                          modified=modified, size=size, gz_size=gz_size)


if __name__ == '__main__':
    prefix = '/mnt/lustre/scratch/inf/mmb28/FeatureExtractionToolkit'

    _random_baselines()

    _count_vectors_gigaw()
    _turian_vectors()

    _glove_vectors_wiki()
    for percent in [1] + list(range(10, 101, 10)):
        _do_w2v_vectors(percent)

    for percent in [15, 50] + [1, 10, 20, 30, 40, 60, 70, 80, 90, 100]:
        _do_w2v_vectors(percent, unlabelled='wiki')

    _ppmi_vectors([10], ['gigaw'])
    _ppmi_vectors([11], ['wikipedia'])

    # verify vectors have been included just once
    vectors = []
    for v in db.Vectors.select():
        data = v._data
        del data['id']
        del data['modified']
        del data['size']
        del data['gz_size']
        vectors.append('_'.join(str(data[k]) for k in sorted(data.keys())))
    if len(set(vectors)) != len(vectors):
        raise ValueError('Duplicated vectors have been entered into database')

    # check the wrong corpus name is not contained in path
    for v in db.Vectors.select():
        if v.unlabelled:
            if v.unlabelled.startswith('wiki'):
                assert 'giga' not in v.path
            if v.unlabelled.startswith('giga'):
                assert 'wiki' not in v.path