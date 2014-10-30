from glob import glob
import os
import shutil
import sys
import time
from itertools import chain

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
from discoutils.misc import Bunch
from thesisgenerator.composers.vectorstore import *
from thesisgenerator.utils.conf_file_utils import set_in_conf_file
from thesisgenerator.utils import db

'''
Once all thesauri with ngram entries (obtained using different composition methods) have been built offline,
use this script to generate the conf files required to run them through the classification framework
'''


def vectors_from_settings(unlab_name, algorithm, composer_name, svd_dims, percent=100):
    v = db.Vectors.select().where((db.Vectors.dimensionality == svd_dims) &
                                  (db.Vectors.unlabelled == unlab_name) &
                                  (db.Vectors.composer == composer_name) &
                                  (db.Vectors.algorithm == algorithm) &
                                  (db.Vectors.unlabelled_percentage == percent))
    return v[0]


def window_vector_settings():
    unlab = 'gigaw'
    algo = 'count_windows'
    composer_algos = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer,
                      RightmostWordComposer, BaroniComposer, Bunch(name='Observed')]
    for c in composer_algos:
        for svd_dims in [0, 100]:
            if svd_dims == 0 and c == BaroniComposer:
                continue  # Baroni needs SVD
            yield unlab, algo, c.name, svd_dims


def dependency_vector_settings():
    unlab = 'gigaw'
    algo = 'count_dependencies'
    # can't easily run Julie's observed dependency code, ignore it
    composer_algos = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer, RightmostWordComposer]
    for c in composer_algos:
        for svd_dims in [0, 100]:
            yield unlab, algo, c.name, svd_dims


def turian_vector_settings():
    unlab = 'turian'
    algo = 'turian'
    composer_algos = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer,
                      RightmostWordComposer, Bunch(name='Socher')]
    for c in composer_algos:
        yield unlab, algo, c.name, 100


def word2vec_vector_settings():
    unlab = 'gigaw'
    algo = 'word2vec'
    composer_algos = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer, RightmostWordComposer]
    for c in composer_algos:
        yield unlab, algo, c.name, 100


def all_vector_settings():
    yield from dependency_vector_settings()
    yield from window_vector_settings()
    yield from turian_vector_settings()
    yield from word2vec_vector_settings()


def baselines():
    # random-neighbour experiments. These include an "random_neighbour_thesaurus=True" option in the conf file
    random_vectors = db.Vectors.get(db.Vectors.algorithm == 'random')
    for corpus in all_corpora:
        e = db.ClassificationExperiment(labelled=corpus, vectors=random_vectors)
        experiments.append(e)

        # signifier experiments (bag-of-words)
        e = db.ClassificationExperiment(labelled=corpus, decode_handler='BaseFeatureHandler')
        experiments.append(e)


def all_standard_experiments():
    for s in all_vector_settings():
        for labelled_corpus in all_corpora:
            e = db.ClassificationExperiment(labelled=labelled_corpus, vectors=vectors_from_settings(*s))
            experiments.append(e)


def hybrid_experiments():
    handler = 'SignifierSignifiedFeatureHandler'

    for s in chain(word2vec_vector_settings(), turian_vector_settings()):
        e = db.ClassificationExperiment(labelled=r2_corpus, vectors=vectors_from_settings(*s),
                                        decode_handler=handler)
        experiments.append(e)


def use_similarity_experiments():
    for s in chain(word2vec_vector_settings(), turian_vector_settings()):
        e = db.ClassificationExperiment(labelled=r2_corpus, vectors=vectors_from_settings(*s),
                                        use_similarity=True)
        experiments.append(e)


def an_only_nn_only_experiments_r2():
    for feature_type in ['AN', 'NN']:
        for s in chain(word2vec_vector_settings(), turian_vector_settings()):
            e = db.ClassificationExperiment(labelled=r2_corpus, vectors=vectors_from_settings(*s),
                                            document_features=feature_type)
            experiments.append(e)


def word2vec_with_less_data_on_r2():
    for unlab, algo, composer, svd_dims in word2vec_vector_settings():
        # only up to 90%, 100% was done separately above
        for percent in range(10, 91, 10):
            e = db.ClassificationExperiment(labelled=r2_corpus,
                                            vectors=vectors_from_settings(unlab, algo, composer,
                                                                          svd_dims, percent))
            experiments.append(e)

prefix = '/mnt/lustre/scratch/inf/mmb28/thesisgenerator/sample-data'
techtc_corpora = list(os.path.join(*x.split(os.sep)[-2:]) \
                      for x in glob('%s/techtc100-clean/*' % prefix) \
                      if not x.endswith('.gz'))
r2_corpus = 'reuters21578/r8-tagged-grouped'
mr_corpus = 'movie-reviews-tagged'
am_corpus = 'amazon_grouped-tagged'
all_corpora = techtc_corpora + [r2_corpus, mr_corpus, am_corpus]

db.ClassificationExperiment.raw('TRUNCATE TABLE `classificationexperiment`;')
experiments = []
baselines()
all_standard_experiments()
hybrid_experiments()
use_similarity_experiments()
an_only_nn_only_experiments_r2()
word2vec_with_less_data_on_r2()

# re-order experiments so that the hard ones (high-memory, long-running) come first
def _myorder(item):
    """
    If the thing is AM move it to the from of the sorted list, sort
    all other items lexicographically
    :param item:
    :return:
    """
    data = getattr(item, 'labelled')
    if 'amazon' in data:
        return chr(1)
    return data


sorted_experiments = sorted(experiments, key=_myorder)
experiments = []
for new_id, e in enumerate(sorted_experiments, 1):
    e.id = new_id
    experiments.append(e)
for e in experiments:
    e.save(force_insert=True)
    print('%s,' % e)

# sys.exit(0)
print('Writing conf files')
megasuperbase_conf_file = 'conf/exp1-superbase.conf'
for exp in experiments:
    if exp.id % 20 == 0:
        print('Writing exp %d' % exp.id)
    # sanity check
    experiment_dir = 'conf/exp%d' % exp.id
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)

    base_conf_file = os.path.join(experiment_dir, 'exp%d_base.conf' % exp.id)
    shutil.copy(megasuperbase_conf_file, base_conf_file)

    if exp.vectors is None and 'Base' in exp.decode_handler:
        # signifier baseline, not using a thesaurus, so shouldn't do any feature selection based on the thesaurus
        set_in_conf_file(base_conf_file, ['feature_selection', 'must_be_in_thesaurus'], False)

    set_in_conf_file(base_conf_file, ['training_data'], os.path.join(prefix, exp.labelled))
    set_in_conf_file(base_conf_file, ['feature_extraction', 'decode_token_handler'],
                     'thesisgenerator.plugins.bov_feature_handlers.%s' % exp.decode_handler)
    set_in_conf_file(base_conf_file, ['feature_extraction', 'random_neighbour_thesaurus'],
                     exp.vectors is not None and exp.vectors.algorithm == 'random')
    set_in_conf_file(base_conf_file, ['vector_sources', 'neighbours_file'],
                     exp.vectors.path if exp.vectors else '')
    set_in_conf_file(base_conf_file, ['output_dir'], './conf/exp%d/output' % exp.id)
    set_in_conf_file(base_conf_file, ['name'], 'exp%d' % exp.id)

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