from functools import wraps
from glob import glob
from collections import Counter
import sys
from itertools import chain

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
from discoutils.misc import Bunch, mkdirs_if_not_exists
from thesisgenerator.composers.vectorstore import *
from thesisgenerator.utils.conf_file_utils import parse_config_file
from thesisgenerator.utils import db


def printing_decorator(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        print('Before function %s: %d experiments' % (f.__name__, len(experiments)))
        return f(*args, **kwds)

    return wrapper


'''
Once all thesauri with ngram entries (obtained using different composition methods) have been built offline,
use this script to generate the conf files required to run them through the classification framework
'''


def vectors_from_settings(unlab_name, algorithm, composer_name, svd_dims, percent=100, rep=0, ppmi=False):
    v = db.Vectors.select().where((db.Vectors.dimensionality == svd_dims) &
                                  (db.Vectors.unlabelled == unlab_name) &
                                  (db.Vectors.composer == composer_name) &
                                  (db.Vectors.algorithm == algorithm) &
                                  (db.Vectors.rep == rep) &
                                  (db.Vectors.use_ppmi == ppmi))
    # peewee cant easily do selects that contain checks of float values
    # lets do a post-filter
    results = [res for res in v if abs(res.unlabelled_percentage - percent) < 1e-6]
    assert len(results) == 1
    return list(results)[0]


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


def word2vec_vector_settings(unlab='gigaw'):
    algo = 'word2vec'
    composer_algos = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer, RightmostWordComposer]
    for c in composer_algos:
        yield unlab, algo, c.name, 100


def glove_vector_settings(unlab='wiki'):
    algo = 'glove'
    composer_algos = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer, RightmostWordComposer]
    for c in composer_algos:
        yield unlab, algo, c.name, 100


def all_vector_settings():
    yield from dependency_vector_settings()
    yield from window_vector_settings()
    yield from turian_vector_settings()
    yield from word2vec_vector_settings()


@printing_decorator
def baselines(corpora=None):
    # random-neighbour experiments. These include an "random_neighbour_thesaurus=True" option in the conf file
    if not corpora:
        corpora = all_corpora
    random_neigh = db.Vectors.get(db.Vectors.algorithm == 'random_neigh')
    for corpus in corpora:
        e = db.ClassificationExperiment(labelled=corpus, vectors=random_neigh)
        experiments.append(e)

        # signifier experiments (bag-of-words)
        e = db.ClassificationExperiment(labelled=corpus, decode_handler='BaseFeatureHandler')
        experiments.append(e)


@printing_decorator
def all_standard_experiments(corpora=None):
    if not corpora:
        corpora = all_corpora
    for s in all_vector_settings():  # yields 28 times
        for labelled_corpus in corpora:
            e = db.ClassificationExperiment(labelled=labelled_corpus, vectors=vectors_from_settings(*s))
            experiments.append(e)


@printing_decorator
def hybrid_experiments_r2_amazon_turian_word2vec():
    handler = 'SignifierSignifiedFeatureHandler'

    for s in chain(word2vec_vector_settings(), turian_vector_settings()):
        for labelled in [r2_corpus, am_corpus]:
            e = db.ClassificationExperiment(labelled=labelled,
                                            vectors=vectors_from_settings(*s),
                                            decode_handler=handler)
            experiments.append(e)


@printing_decorator
def an_only_nn_only_experiments_r2():
    for feature_type in ['AN', 'NN']:
        for s in chain(word2vec_vector_settings(), turian_vector_settings()):
            e = db.ClassificationExperiment(labelled=r2_corpus, vectors=vectors_from_settings(*s),
                                            document_features=feature_type)
            experiments.append(e)


@printing_decorator
def word2vec_repeated_runs_on_amazon():
    labelled = am_corpus
    for unlab, algo, composer, svd_dims in word2vec_vector_settings():
        for rep in [-1, 1, 2]:  # 0 done as part of "standard experiments"
            # only the second and third run of word2vec on the entire data set, the first was done above
            e = db.ClassificationExperiment(labelled=labelled,
                                            vectors=vectors_from_settings(unlab, algo, composer,
                                                                          svd_dims, rep=rep))
            experiments.append(e)


@printing_decorator
def glove_vectors_amazon():
    for s in glove_vector_settings():
        e = db.ClassificationExperiment(labelled=am_corpus, vectors=vectors_from_settings(*s))
        experiments.append(e)


@printing_decorator
def random_vectors(corpus):
    random_vect = db.Vectors.get(db.Vectors.algorithm == 'random_vect')
    e = db.ClassificationExperiment(labelled=corpus, vectors=random_vect)
    experiments.append(e)


@printing_decorator
def w2v_learning_curve_amazon(unlab='gigaw', percent=[1, 10, 20, 30, 40, 50, 60, 70, 80, 90]):
    for settings in word2vec_vector_settings(unlab):
        # only up to 90% to avoid dupliting w2v-gigaw-100% (part of "standard experiments")
        for p in percent:
            e = db.ClassificationExperiment(labelled=am_corpus,
                                            vectors=vectors_from_settings(*settings, percent=p))
            experiments.append(e)


@printing_decorator
def varying_k_with_w2v_on_r2():
    for k in [1, 5]:
        for settings in word2vec_vector_settings():
            e = db.ClassificationExperiment(labelled=r2_corpus, vectors=vectors_from_settings(*settings),
                                            k=k)
            experiments.append(e)


@printing_decorator
def different_neighbour_strategies_r2():
    strat = 'skipping'
    for settings in word2vec_vector_settings():
        e = db.ClassificationExperiment(labelled=r2_corpus, vectors=vectors_from_settings(*settings),
                                        neighbour_strategy=strat)
        experiments.append(e)


@printing_decorator
def wikipedia_w2v_amazon():
    unlab = 'wiki'
    for p in [15, 50]:
        for _, algo, composer_name, dims in word2vec_vector_settings():
            e = db.ClassificationExperiment(labelled=am_corpus, vectors=vectors_from_settings(unlab,
                                                                                              algo,
                                                                                              composer_name,
                                                                                              dims,
                                                                                              percent=p))
            experiments.append(e)


@printing_decorator
def corrupted_w2v_amazon():
    for noise in np.arange(.2, 2.1, .2):
        v = vectors_from_settings('wiki', 'word2vec', 'Add', 100, percent=100)
        e = db.ClassificationExperiment(labelled=am_corpus, vectors=v, noise=noise)
        experiments.append(e)


@printing_decorator
def count_with_ppmi_no_svd_amazon(corpus=None):
    if not corpus:
        corpus = am_corpus

    for composer in [AdditiveComposer, MultiplicativeComposer,
                     LeftmostWordComposer, RightmostWordComposer]:
        for algo in ['count_dependencies', 'count_windows']:
            v = vectors_from_settings('gigaw', algo, composer.name, svd_dims=0, ppmi=True)
            e = db.ClassificationExperiment(labelled=corpus, vectors=v)
            experiments.append(e)

@printing_decorator
def count_wiki_with_ppmi_no_svd_amazon():
    for algo in ['count_windows', 'count_dependencies']:
        for composer in [AdditiveComposer, MultiplicativeComposer,
                         LeftmostWordComposer, RightmostWordComposer]:
            v = vectors_from_settings('wikipedia', algo, composer.name, svd_dims=0, ppmi=True)
            e = db.ClassificationExperiment(labelled=am_corpus, vectors=v)
            experiments.append(e)


if __name__ == '__main__':
    prefix = '/mnt/lustre/scratch/inf/mmb28/thesisgenerator/sample-data'
    techtc_corpora = sorted(list(os.path.join(*x.split(os.sep)[-2:]) \
                                 for x in glob('%s/techtc100-clean/*' % prefix) \
                                 if not x.endswith('.gz')))
    r2_corpus = 'reuters21578/r8-tagged-grouped'
    mr_corpus = 'movie-reviews-tagged'
    am_corpus = 'amazon_grouped-tagged'
    maas_corpus = 'aclImdb-tagged'
    # havent added maas to all_corpora to avoid changing the ids of long running amazon jobs
    all_corpora = techtc_corpora + [r2_corpus, mr_corpus, am_corpus]

    db.ClassificationExperiment.raw('TRUNCATE TABLE `classificationexperiment`;')
    experiments = []
    baselines()
    all_standard_experiments()
    hybrid_experiments_r2_amazon_turian_word2vec()
    word2vec_repeated_runs_on_amazon()
    random_vectors(r2_corpus)
    w2v_learning_curve_amazon()
    varying_k_with_w2v_on_r2()
    random_vectors(am_corpus)
    # wikipedia experiments on amazon
    wikipedia_w2v_amazon()
    # maas IMDB sentiment experiments
    # baselines(corpora=[maas_corpus])
    # random_vectors(maas_corpus)
    # all_standard_experiments(corpora=[maas_corpus])
    # other more recent stuff
    corrupted_w2v_amazon()
    count_with_ppmi_no_svd_amazon()
    glove_vectors_amazon()
    # 15, 50% done as a part of wikipedia_w2v_amazon()
    w2v_learning_curve_amazon(unlab='wiki', percent=[1, 10, 20, 30, 40, 60, 70, 80, 90, 100])
    count_wiki_with_ppmi_no_svd_amazon()

    # various other experiments that aren't as interesting
    # an_only_nn_only_experiments_r2()
    # different_neighbour_strategies()
    print('Total experiments: %d' % len(experiments))


    # re-order experiments so that the hard ones (high-memory, long-running) come first
    def _myorder(item):
        """
        If the thing is AM/Maas move it to the from of the sorted list, leave
        all other items in the order they were in
        :param item:
        :return:
        """
        lab = getattr(item[1], 'labelled')
        if 'amazon' in lab or 'aclImdb' in lab:
            return chr(1), item[0]
        return 'whatever', item[0]

    # assign ID's starting from 500 to all fast experiments
    # I can now add slow experiments at will without messing up the order of the slow ones
    # e.g. experiment numbers will be 1, 2, 3..., 75, 500, 501, ...
    # which group into slow and fast (1, 2, 3..., 75), (500, 501, ...)
    # this is a little buggy in that the first fast exp is in the slow group
    # i.e. (1, 2, 3..., 74), (75, 500, 501, ...)
    # but whatever, not worth my time
    sorted_experiments = sorted(enumerate(experiments), key=_myorder)
    experiments = []
    new_id = 1
    prev_exp = None
    for old_id, e in sorted_experiments:
        # print('%d --> %d' % (old_id, new_id))
        e.id = new_id
        experiments.append(e)
        new_id += 1
        if hasattr(prev_exp, 'labelled') and \
                        prev_exp.labelled in [am_corpus, maas_corpus] and \
                        e.labelled not in [am_corpus, maas_corpus]:
            new_id = 500
        prev_exp = e

    # sys.exit(0)
    for e in experiments:
        e.save(force_insert=True)

        # verify experiments aren't being duplicated
    if len(set(experiments)) != len(experiments):
        raise ValueError('Duplicated experiments exist: %s' % Counter(experiments).most_common(5))

    # sys.exit(0)
    print('Writing conf files')
    megasuperbase_conf_file = 'conf/exp1-superbase.conf'
    for exp in experiments:
        if exp.id % 50 == 0:
            print('Writing exp %d' % exp.id)
        # sanity check
        experiment_dir = 'conf/exp%d' % exp.id
        mkdirs_if_not_exists(experiment_dir)

        base_conf_file = os.path.join(experiment_dir, 'exp%d_base.conf' % exp.id)
        conf, _ = parse_config_file(megasuperbase_conf_file)

        if exp.vectors is None and 'Base' in exp.decode_handler:
            # signifier baseline, not using a thesaurus, so shouldn't do any feature selection based on the thesaurus
            conf['feature_selection']['must_be_in_thesaurus'] = False

        conf['training_data'] = os.path.join(prefix, exp.labelled)
        conf['feature_extraction']['decode_token_handler'] = \
            'thesisgenerator.plugins.bov_feature_handlers.%s' % exp.decode_handler
        conf['feature_extraction']['random_neighbour_thesaurus'] = \
            exp.vectors is not None and exp.vectors.algorithm == 'random_neigh'
        conf['vector_sources']['neighbours_file'] = exp.vectors.path if exp.vectors else ''
        conf['vector_sources']['noise'] = exp.noise
        conf['output_dir'] = './conf/exp%d/output' % exp.id
        conf['name'] = 'exp%d' % exp.id
        conf['feature_extraction']['k'] = exp.k
        requested_features = exp.document_features.split('_')
        for doc_feature_type in ['AN', 'NN', 'VO', 'SVO']:
            conf['feature_extraction'][
                'extract_%s_features' % doc_feature_type] = doc_feature_type in requested_features

        # do not allow lexical overlap to prevent Left and Right from relying on word identity
        conf['vector_sources']['allow_lexical_overlap'] = False
        conf['vector_sources']['neighbour_strategy'] = exp.neighbour_strategy

        if exp.use_similarity:
            conf['feature_extraction']['sim_compressor'] = 'thesisgenerator.utils.misc.unit'
        else:
            conf['feature_extraction']['sim_compressor'] = 'thesisgenerator.utils.misc.one'

        with open(base_conf_file, 'wb') as inf:
            conf.write(inf)

        # we've just written a conf file to a directory that may have contained results from before
        # check those results were done with the same configuration we just created. This is needed
        # because as we reorder experiments existing results may end up with a different ID
        # otherwise the experiment will have to be re-run
        # only checking some of the important parameters
        previous_conf_file = 'conf/exp{0}/exp{0}_base-variants/exp{0}-0.conf'.format(exp.id)
        if os.path.exists(previous_conf_file):
            old_conf, _ = parse_config_file(previous_conf_file)
            for a, b in [(old_conf['vector_sources']['neighbours_file'],
                          conf['vector_sources']['neighbours_file']),
                         (old_conf['feature_extraction']['decode_token_handler'],
                          conf['feature_extraction']['decode_token_handler']),
                         (old_conf['training_data'], conf['training_data'])]:
                if a != b:
                    print('Exp: %d, was %r, is now %r' % (exp.id, a, b))
