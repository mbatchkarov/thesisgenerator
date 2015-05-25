from functools import wraps
from glob import glob
from collections import Counter
import sys
from itertools import chain
from peewee import IntegrityError

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


def _make_expansions(**kwargs):
    # do not use pw.create_or_get, it doesn't fill in default values for fields
    e = db.Expansions(**kwargs)
    try:
        e.save(force_insert=True)
    except IntegrityError:
        # already in DB, ignore
        pass
    return e


def vectors_by_type(feature_type, composer_name):
    v = db.Vectors.select().where((db.Vectors.contents.contains(feature_type)) &
                                  (db.Vectors.composer == composer_name))
    return list(v)


def vectors_from_settings(unlab_name, algorithm, composer_name, svd_dims, percent=100, rep=0, ppmi=False):
    assert svd_dims > 1 or svd_dims is None
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
                      RightmostWordComposer, BaroniComposer, GuevaraComposer,
                      Bunch(name='Observed')]
    for c in composer_algos:
        for svd_dims in [100]:
            if svd_dims == 0 and c in (BaroniComposer, GuevaraComposer):
                continue  # Baroni/Guevara needs SVD
            yield unlab, algo, c.name, svd_dims


def dependency_vector_settings():
    unlab = 'gigaw'
    algo = 'count_dependencies'
    # can't easily run Julie's observed dependency code, ignore it
    composer_algos = [AdditiveComposer, MultiplicativeComposer,
                      LeftmostWordComposer, RightmostWordComposer]
    for c in composer_algos:
        for svd_dims in [100]:
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
    composer_algos = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer,
                      RightmostWordComposer]
    for c in composer_algos:
        yield unlab, algo, c.name, 100


def glove_vector_settings(unlab='wiki'):
    algo = 'glove'
    composer_algos = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer,
                      RightmostWordComposer]
    for c in composer_algos:
        yield unlab, algo, c.name, 100


def all_vector_settings():
    yield from dependency_vector_settings()
    yield from window_vector_settings()
    yield from turian_vector_settings()
    yield from word2vec_vector_settings()


@printing_decorator
def random_baselines(corpora=None, document_features_tr='J+N+AN+NN', document_features_ev='AN+NN'):
    # random-neighbour and random-vector experiments.
    # The former include an "random_neighbour_thesaurus=True" option in the conf file

    if corpora is None:
        corpora = [am_corpus]
    random_neigh = db.Vectors.get(db.Vectors.algorithm == 'random_neigh')
    random_vect = db.Vectors.get(db.Vectors.algorithm == 'random_vect')

    for corpus in corpora:
        for v in [random_neigh, random_vect]:
            e = db.ClassificationExperiment(labelled=corpus,
                                            expansions=_make_expansions(vectors=v.id),
                                            document_features_tr=document_features_tr,
                                            document_features_ev=document_features_ev)
            experiments.append(e)


@printing_decorator
def all_standard_gigaw_experiments(corpora=None):
    if not corpora:
        corpora = [am_corpus]
    for s in all_vector_settings():  # yields 28 times
        for labelled_corpus in corpora:
            v = vectors_from_settings(*s)
            e = db.ClassificationExperiment(labelled=labelled_corpus,
                                            expansions=_make_expansions(vectors=v))
            experiments.append(e)


@printing_decorator
def hybrid_experiments_turian_word2vec_gigaw():
    h = 'SignifierSignifiedFeatureHandler'

    for s in chain(word2vec_vector_settings(unlab='wiki'), turian_vector_settings()):
        v = vectors_from_settings(*s)
        e = db.ClassificationExperiment(labelled=am_corpus,
                                        expansions=_make_expansions(vectors=v,
                                                                    decode_handler=h))
        experiments.append(e)


@printing_decorator
def an_only_nn_only_experiments_amazon():
    for feature_type in ['AN', 'NN']:
        for s in word2vec_vector_settings():
            v = vectors_from_settings(*s)
            e = db.ClassificationExperiment(expansions=_make_expansions(vectors=v),
                                            labelled=am_corpus,
                                            document_features_ev=feature_type)
            experiments.append(e)


@printing_decorator
def glove_vectors_amazon():
    for s in glove_vector_settings():
        v = vectors_from_settings(*s)
        e = db.ClassificationExperiment(labelled=am_corpus, expansions=_make_expansions(vectors=v))
        experiments.append(e)


@printing_decorator
def nondistributional_baselines(corpora=None, document_features_tr='J+N+AN+NN', document_features_ev='AN+NN'):
    # signifier experiments (bag-of-words)
    if corpora is None:
        corpora = [am_corpus]
    for corpus in corpora:
        e = db.ClassificationExperiment(labelled=corpus,
                                        document_features_tr=document_features_tr,
                                        document_features_ev=document_features_ev,
                                        expansions=_make_expansions(decode_handler='BaseFeatureHandler'))
        experiments.append(e)


@printing_decorator
def w2v_learning_curve_amazon(unlab='wiki', percent=[1, 10, 20, 30, 40, 50, 60, 70, 80, 90]):
    for settings in word2vec_vector_settings(unlab):
        # only up to 90% to avoid duplicating w2v-gigaw-100% (part of "standard experiments")
        for p in percent:
            v = vectors_from_settings(*settings, percent=p)
            e = db.ClassificationExperiment(labelled=am_corpus, expansions=_make_expansions(vectors=v))
            experiments.append(e)


@printing_decorator
def varying_k_with_w2v_on_amazon():
    for k in [1, 5]:
        for settings in word2vec_vector_settings(unlab='wiki'):
            v = vectors_from_settings(*settings)
            e = db.ClassificationExperiment(labelled=am_corpus,
                                            expansions=_make_expansions(vectors=v,
                                                                        k=k))
            experiments.append(e)


@printing_decorator
def different_neighbour_strategies_r2():
    strat = 'skipping'
    for settings in word2vec_vector_settings():
        v = vectors_from_settings(*settings)
        e = db.ClassificationExperiment(labelled=r2_corpus,
                                        expansions=_make_expansions(vectors=v,
                                                                    neighbour_strategy=strat))
        experiments.append(e)


@printing_decorator
def initial_wikipedia_w2v_amazon_with_repeats():
    unlab = 'wiki'
    for p in [15, 50]:
        for rep in [-1, 0, 1, 2]:
            for _, algo, composer_name, dims in word2vec_vector_settings():
                v = vectors_from_settings(unlab, algo, composer_name, dims, percent=p, rep=rep)
                e = db.ClassificationExperiment(labelled=am_corpus, expansions=_make_expansions(vectors=v))
                experiments.append(e)


@printing_decorator
def corrupted_w2v_wiki_amazon():
    for noise in np.arange(.2, 2.1, .2):
        v = vectors_from_settings('wiki', 'word2vec', 'Add', 100, percent=100)
        e = db.ClassificationExperiment(labelled=am_corpus,
                                        expansions=_make_expansions(vectors=v,
                                                                    noise=noise))
        experiments.append(e)


@printing_decorator
def count_wiki_with_svd_no_ppmi_amazon():
    composers = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer, RightmostWordComposer]
    for algo in ['count_dependencies', 'count_windows']:
        if 'windows' in algo:
            composers.append(Bunch(name='Observed'))
        for composer in composers:
            v = vectors_from_settings('wiki', algo, composer.name, svd_dims=100, ppmi=False)
            e = db.ClassificationExperiment(labelled=am_corpus, expansions=_make_expansions(vectors=v))
            experiments.append(e)


@printing_decorator
def equalised_coverage_experiments():
    # w2v with 4 composers (high-coverage model) with coverage of corresponding count windows (low-coverage)
    # CW add (high-coverage) with coverage of CW baroni (low-coverage)
    algo = 'count_windows'
    composers = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer, RightmostWordComposer]
    for composer in composers:
        regular_vect = vectors_from_settings('wiki', 'word2vec', composer.name, 100)
        entries_of = vectors_from_settings('wiki', 'count_windows', composer.name, 100)
        e = db.ClassificationExperiment(labelled=am_corpus,
                                        expansions=_make_expansions(vectors=regular_vect,
                                                                    entries_of=entries_of))
        experiments.append(e)

        regular_vect = vectors_from_settings('wiki', 'count_windows', composer.name, 100)
        entries_of = vectors_from_settings('wiki', 'count_windows', 'Baroni', 100)
        e = db.ClassificationExperiment(labelled=am_corpus,
                                        expansions=_make_expansions(vectors=regular_vect,
                                                                    entries_of=entries_of))
        experiments.append(e)


@printing_decorator
def with_unigrams_at_decode_time():
    composers = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer, RightmostWordComposer]
    for composer in composers:
        vect = [vectors_from_settings('gigaw', 'word2vec', composer.name, svd_dims=100, percent=100),
                vectors_from_settings('wiki', 'word2vec', composer.name, svd_dims=100, percent=15),
                vectors_from_settings('wiki', 'word2vec', composer.name, svd_dims=100, percent=100),
                ]
        for v in vect:
            e = db.ClassificationExperiment(expansions=_make_expansions(vectors=v),
                                            labelled=am_corpus,
                                            document_features_ev='J+N+AN+NN')
            experiments.append(e)


@printing_decorator
def with_lexical_overlap_and_unigrams_at_decode_time():
    composers = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer, RightmostWordComposer]
    for composer in composers:
        vect = [vectors_from_settings('gigaw', 'word2vec', composer.name, svd_dims=100, percent=100),
                vectors_from_settings('wiki', 'word2vec', composer.name, svd_dims=100, percent=15),
                vectors_from_settings('wiki', 'word2vec', composer.name, svd_dims=100, percent=100),
                ]
        for v in vect:
            e = db.ClassificationExperiment(expansions=_make_expansions(vectors=v, allow_overlap=True),
                                            labelled=am_corpus,
                                            document_features_ev='J+N+AN+NN')
            experiments.append(e)


@printing_decorator
def verb_phrases_svo(document_features_tr, document_features_ev, allow_overlap=True):
    composers = [AdditiveComposer, MultiplicativeComposer,
                 GrefenstetteMultistepComposer, VerbComposer, CopyObject]
    for comp in composers:
        vect = vectors_by_type('SVO', comp.name)
        for v in vect:
            e = db.ClassificationExperiment(labelled=am_corpus,
                                            document_features_tr=document_features_tr,
                                            document_features_ev=document_features_ev,
                                            expansions=_make_expansions(vectors=v,
                                                                        allow_overlap=allow_overlap))
            experiments.append(e)


@printing_decorator
def kmeans_experiments():
    for cl in db.Clusters.select():
        e = db.ClassificationExperiment(labelled=am_corpus, clusters=cl)
        experiments.append(e)


def write_conf_files():
    global experiments
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
        dupl = [x for x in experiments if x.__hash__() == Counter(experiments).most_common(1)[0][0].__hash__()]
        for x in dupl:
            print(x._key(), '\n', x.expansions._key(), x.clusters)
        raise ValueError('Duplicated experiments exist: %s' % dupl)

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

        # options for all experiments
        conf['training_data'] = os.path.join(prefix, exp.labelled)
        conf['output_dir'] = './conf/exp%d/output' % exp.id
        conf['name'] = 'exp%d' % exp.id
        for time, requested_features in zip(['train', 'decode'],
                                            [exp.document_features_tr, exp.document_features_ev]):
            requested_features = requested_features.split('+')

            unigram_feats = sorted([foo for foo in requested_features if len(foo) == 1])
            conf['feature_extraction']['%s_time_opts' % time]['extract_unigram_features'] = unigram_feats

            phrasal_feats = sorted([foo for foo in requested_features if len(foo) > 1])
            conf['feature_extraction']['%s_time_opts' % time]['extract_phrase_features'] = phrasal_feats

        # options for experiments where we use feature expansion
        if exp.expansions:
            if exp.expansions.vectors is None and 'Base' in exp.expansions.decode_handler:
                # signifier baseline, not using a thesaurus, so shouldn't do any feature
                # selection based on the thesaurus
                conf['feature_selection']['must_be_in_thesaurus'] = False

            conf['feature_extraction']['decode_token_handler'] = \
                'thesisgenerator.plugins.bov_feature_handlers.%s' % exp.expansions.decode_handler
            conf['feature_extraction']['random_neighbour_thesaurus'] = \
                exp.expansions.vectors is not None and exp.expansions.vectors.algorithm == 'random_neigh'
            conf['vector_sources']['neighbours_file'] = exp.expansions.vectors.path if exp.expansions.vectors else ''
            conf['vector_sources']['noise'] = exp.expansions.noise
            conf['feature_extraction']['k'] = exp.expansions.k



            # do not allow lexical overlap to prevent Left and Right from relying on word identity
            conf['vector_sources']['allow_lexical_overlap'] = exp.expansions.allow_overlap
            conf['vector_sources']['neighbour_strategy'] = exp.expansions.neighbour_strategy
            conf['vector_sources']['entries_of'] = exp.expansions.entries_of.path if exp.expansions.entries_of else ''

            if exp.expansions.use_similarity:
                conf['feature_extraction']['sim_compressor'] = 'thesisgenerator.utils.misc.unit'
            else:
                conf['feature_extraction']['sim_compressor'] = 'thesisgenerator.utils.misc.one'

        # options for experiments where we use clustered disco features
        if exp.clusters:
            conf['feature_extraction']['vectorizer'] = \
                'thesisgenerator.plugins.kmeans_disco.KmeansVectorizer'
            conf['vector_sources']['clusters_file'] = exp.clusters.path

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

    # db.ClassificationExperiment.raw('TRUNCATE TABLE `classificationexperiment`;')
    experiments = []

    ################################################################
    # NOUN PHRASES
    ################################################################
    random_baselines()
    nondistributional_baselines()
    nondistributional_baselines(document_features_ev='J+N+AN+NN')

    all_standard_gigaw_experiments()
    hybrid_experiments_turian_word2vec_gigaw()
    varying_k_with_w2v_on_amazon()
    # wikipedia experiments on amazon
    initial_wikipedia_w2v_amazon_with_repeats()
    # maas IMDB sentiment experiments
    # baselines(corpora=[maas_corpus])
    # random_vectors(maas_corpus)
    # all_standard_experiments(corpora=[maas_corpus])
    # other more recent stuff
    corrupted_w2v_wiki_amazon()
    glove_vectors_amazon()
    # 15, 50% done as a part of initial_wikipedia_w2v_amazon()
    w2v_learning_curve_amazon(percent=[1, 10, 20, 30, 40, 60, 70, 80, 90, 100])
    # PPMI-no-SVD ones take days to classify, let's used SVD ones instead
    # currently can't do PPMI + SVD, and it probably doesn't make sense
    count_wiki_with_svd_no_ppmi_amazon()
    an_only_nn_only_experiments_amazon()
    equalised_coverage_experiments()
    with_unigrams_at_decode_time()
    with_lexical_overlap_and_unigrams_at_decode_time()

    ################################################################
    # VERB PHRASES
    ################################################################

    # pad with J+N to increase performance a wee bit
    nondistributional_baselines(document_features_tr='J+N+V',
                                document_features_ev='J+N+V+SVO')
    verb_phrases_svo(document_features_tr='J+N+V',
                     document_features_ev='J+N+V+SVO',
                     allow_overlap=True)

    kmeans_experiments()
    # various other experiments that aren't as interesting
    # different_neighbour_strategies() # this takes a long time


    print('Total experiments: %d' % len(experiments))

    write_conf_files()
