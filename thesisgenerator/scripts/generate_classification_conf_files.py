from functools import wraps
from glob import glob
from collections import Counter
import sys
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
        print('Before function %s: %d experiments' % (f.__name__, len(list(db.ClassificationExperiment.select()))))
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


def vectors_from_settings(unlab_name, algorithm, composer_name, svd_dims, percent=100, rep=0,
                          avg=False, reorder=False):
    assert svd_dims > 1 or svd_dims is None
    v = db.Vectors.select().where((db.Vectors.dimensionality == svd_dims) &
                                  (db.Vectors.unlabelled == unlab_name) &
                                  (db.Vectors.composer == composer_name) &
                                  (db.Vectors.algorithm == algorithm) &
                                  (db.Vectors.rep == rep) &
                                  (db.Vectors.avg == avg) &
                                  (db.Vectors.reorder == reorder)
                                  )
    # peewee cant easily do selects that contain checks of float values
    # lets do a post-filter
    results = [res for res in v if abs(res.unlabelled_percentage - percent) < 1e-6]
    assert len(results) == 1, 'Expected unique vectors id, found %d' % len(results)
    return results[0]


def clusters_from_settings(vectors, num_clusters, noise=0):
    v = db.Clusters.select().where((db.Clusters.vectors == vectors) &
                                   (db.Clusters.num_clusters == num_clusters)
                                   )
    # peewee cant easily do selects that contain checks of float values
    # lets do a post-filter
    results = [res for res in v if abs(res.noise - noise) < 1e-6]
    assert len(results) == 1
    return results[0]


def window_vector_settings(unlab='gigaw'):
    algo = 'count_windows'
    composer_algos = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer,
                      RightmostWordComposer, BaroniComposer, GuevaraComposer,
                      Bunch(name='Observed')]
    for c in composer_algos:
        for svd_dims in [100]:
            if svd_dims == 0 and c in (BaroniComposer, GuevaraComposer):
                continue  # Baroni/Guevara needs SVD
            yield unlab, algo, c.name, svd_dims


def dependency_vector_settings(unlab='gigaw'):
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


def all_vector_settings(unlab=['gigaw', 'wiki']):
    for u in unlab:
        yield from dependency_vector_settings(unlab=u)
        yield from window_vector_settings(unlab=u)
        yield from word2vec_vector_settings(unlab=u)

    yield from turian_vector_settings()


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
            e.save(force_insert=True)


@printing_decorator
def all_standard_experiments(labelled=None):
    if not labelled:
        labelled = [am_corpus]
    for s in all_vector_settings():  # yields 35 times
        for labelled_corpus in labelled:
            v = vectors_from_settings(*s)
            e = db.ClassificationExperiment(labelled=labelled_corpus,
                                            expansions=_make_expansions(vectors=v))
            e.save(force_insert=True)


@printing_decorator
def hybrid_experiments_word2vec_gigaw():
    h = 'SignifierSignifiedFeatureHandler'

    for s in word2vec_vector_settings(unlab='wiki'):
        v = vectors_from_settings(*s)
        e = db.ClassificationExperiment(labelled=am_corpus,
                                        expansions=_make_expansions(vectors=v,
                                                                    decode_handler=h))
        e.save(force_insert=True)


@printing_decorator
def an_only_nn_only_experiments_amazon():
    for feature_type in ['AN', 'NN']:
        for s in word2vec_vector_settings():
            v = vectors_from_settings(*s)
            e = db.ClassificationExperiment(expansions=_make_expansions(vectors=v),
                                            labelled=am_corpus,
                                            document_features_ev=feature_type)
            e.save(force_insert=True)


@printing_decorator
def glove_vectors_amazon():
    for s in glove_vector_settings():
        v = vectors_from_settings(*s)
        e = db.ClassificationExperiment(labelled=am_corpus, expansions=_make_expansions(vectors=v))
        e.save(force_insert=True)


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
        e.save(force_insert=True)


@printing_decorator
def w2v_wiki_learning_curve(percent, unlab='wiki', corpus=None, k=3):
    if corpus is None:
        corpus = am_corpus
    for settings in word2vec_vector_settings(unlab):
        for p in percent:
            v = vectors_from_settings(*settings, percent=p)
            e = db.ClassificationExperiment(labelled=corpus, expansions=_make_expansions(vectors=v, k=k))
            e.save(force_insert=True)


@printing_decorator
def varying_k_with_w2v_on_amazon(ks=[1, 10, 30, 50, 75, 100]):
    for k in ks:  # k=3 is a "standard" experiment
        for settings in word2vec_vector_settings(unlab='wiki'):
            v = vectors_from_settings(*settings)
            e = db.ClassificationExperiment(labelled=am_corpus,
                                            expansions=_make_expansions(vectors=v,
                                                                        k=k))
            e.save(force_insert=True)


@printing_decorator
def different_neighbour_strategies_r2():
    strat = 'skipping'
    for settings in word2vec_vector_settings():
        v = vectors_from_settings(*settings)
        e = db.ClassificationExperiment(labelled=r2_corpus,
                                        expansions=_make_expansions(vectors=v,
                                                                    neighbour_strategy=strat))
        e.save(force_insert=True)


@printing_decorator
def initial_wikipedia_w2v_amazon_with_repeats():
    unlab = 'wiki'
    for rep in [3, 0, 1, 2]:
        avg = (rep == 3)
        for _, algo, composer_name, dims in word2vec_vector_settings():
            v = vectors_from_settings(unlab, algo, composer_name, dims, percent=15, rep=rep, avg=avg)
            e = db.ClassificationExperiment(labelled=am_corpus, expansions=_make_expansions(vectors=v))
            e.save(force_insert=True)


@printing_decorator
def wikipedia_w2v_R2_repeats():
    for rep in [3, 0, 1, 2]:
        avg = (rep == 3)
        v = vectors_from_settings('wiki', 'word2vec', 'Add', 100, percent=15, rep=rep, avg=avg)
        e = db.ClassificationExperiment(labelled=r2_corpus, expansions=_make_expansions(vectors=v))
        e.save(force_insert=True)


@printing_decorator
def corrupted_w2v_wiki(corpus, k=3, include_zero=False):
    for noise in np.arange(0 if include_zero else .2, 2.1, .2):
        v = vectors_from_settings('wiki', 'word2vec', 'Add', 100, percent=100)
        e = db.ClassificationExperiment(labelled=corpus,
                                        expansions=_make_expansions(vectors=v,
                                                                    noise=noise,
                                                                    k=k))
        e.save(force_insert=True)


@printing_decorator
def equalised_coverage_experiments():
    # w2v with 4 composers (high-coverage model) with coverage of corresponding count windows (low-coverage)
    # CW add (high-coverage) with coverage of CW baroni (low-coverage)
    composers = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer, RightmostWordComposer]
    for composer in composers:
        regular_vect = vectors_from_settings('wiki', 'word2vec', composer.name, 100)
        entries_of = vectors_from_settings('wiki', 'count_windows', composer.name, 100)
        e = db.ClassificationExperiment(labelled=am_corpus,
                                        expansions=_make_expansions(vectors=regular_vect,
                                                                    entries_of=entries_of))
        e.save(force_insert=True)

        regular_vect = vectors_from_settings('wiki', 'count_windows', composer.name, 100)
        entries_of = vectors_from_settings('wiki', 'count_windows', 'Baroni', 100)
        e = db.ClassificationExperiment(labelled=am_corpus,
                                        expansions=_make_expansions(vectors=regular_vect,
                                                                    entries_of=entries_of))
        e.save(force_insert=True)

@printing_decorator
def equalised_coverage_experiments_v3():
    # count windows with Add composers (high-coverage model) with coverage of count windows with Guevara composition
    regular_vect = vectors_from_settings('wiki', 'count_windows', AdditiveComposer.name, 100)
    entries_of = vectors_from_settings('wiki', 'count_windows', GuevaraComposer.name, 100)
    e = db.ClassificationExperiment(labelled=am_corpus,
                                    expansions=_make_expansions(vectors=regular_vect,
                                                                entries_of=entries_of))
    e.save(force_insert=True)


    # count windows with Guev composers (medium-coverage model) with coverage of count windows with Baroni composition
    regular_vect = vectors_from_settings('wiki', 'count_windows', GuevaraComposer.name, 100)
    entries_of = vectors_from_settings('wiki', 'count_windows', BaroniComposer.name, 100)
    e = db.ClassificationExperiment(labelled=am_corpus,
                                    expansions=_make_expansions(vectors=regular_vect,
                                                                entries_of=entries_of))
    e.save(force_insert=True)


@printing_decorator
def equalised_coverage_experiments_v2(composers=None, percent_reduce_from=None, percent_reduce_to=15):
    #  MORE EQUALISED COVERAGE EXPERIMENTS- WIKI-100% REDUCED TO WIKI-15%
    if composers is None:
        composers = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer, RightmostWordComposer]
    if percent_reduce_from is None:
        percent_reduce_from = [100]

    for composer in composers:
        entries_of = vectors_from_settings('wiki', 'word2vec', composer.name, 100, percent=percent_reduce_to)
        for p in percent_reduce_from:
            regular_vect = vectors_from_settings('wiki', 'word2vec', composer.name, 100, percent=p)
            e = db.ClassificationExperiment(labelled=am_corpus,
                                            expansions=_make_expansions(vectors=regular_vect,
                                                                        entries_of=entries_of))
            e.save(force_insert=True)


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
            e.save(force_insert=True)


@printing_decorator
def with_lexical_overlap():
    composers = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer, RightmostWordComposer]
    for composer in composers:
        vect = [vectors_from_settings('gigaw', 'word2vec', composer.name, svd_dims=100, percent=100),
                vectors_from_settings('wiki', 'word2vec', composer.name, svd_dims=100, percent=15),
                vectors_from_settings('wiki', 'word2vec', composer.name, svd_dims=100, percent=100)]
        for v in vect:
            e = db.ClassificationExperiment(expansions=_make_expansions(vectors=v, allow_overlap=True),
                                            labelled=am_corpus)
            e.save(force_insert=True)


@printing_decorator
def verb_phrases_svo(composers=None):
    if not composers:
        composers = [AdditiveComposer, MultiplicativeComposer, VerbComposer, CopyObject]
    algos = ['count_windows', 'glove', 'word2vec']
    for composer in composers:
        # w2v vs glove vs count @ wiki 100 (several composers)
        # wiki 15, giga 100 with w2v only (same composers)
        vect = [vectors_from_settings('wiki', algo, composer.name, 100) for algo in algos] + \
               [vectors_from_settings('gigaw', 'word2vec', composer.name, svd_dims=100, percent=100),
                vectors_from_settings('wiki', 'word2vec', composer.name, svd_dims=100, percent=15)]
        for v in vect:
            e = db.ClassificationExperiment(labelled=am_corpus,
                                            document_features_tr='J+N+V+SVO',
                                            document_features_ev='SVO',
                                            expansions=_make_expansions(vectors=v))
            e.save(force_insert=True)


@printing_decorator
def kmeans_experiments(min_id=1, max_id=30, labelled=None):
    if labelled is None:
        labelled = am_corpus
    for cl in db.Clusters.select():
        if min_id <= cl.id <= max_id:
            e = db.ClassificationExperiment(labelled=labelled, clusters=cl)
            e.save(force_insert=True)


@printing_decorator
def multivectors():
    for v in db.Vectors.select():
        if v.reorder:
            e = db.ClassificationExperiment(labelled=am_corpus,
                                            expansions=_make_expansions(vectors=v))
            e.save(force_insert=True)


@printing_decorator
def multivectors_higher_k():
    for v in db.Vectors.select():
        if v.reorder and v.composer == 'Add':
            e = db.ClassificationExperiment(labelled=am_corpus,
                                            expansions=_make_expansions(vectors=v, k=30))
            e.save(force_insert=True)


@printing_decorator
def unigram_only_vary_k(ks=[1, 3, 5, 7, 10, 15, 20, 30, 40, 50, 75, 100]):
    v = vectors_from_settings('wiki', 'word2vec', None, 100)
    for k in ks:
        e = db.ClassificationExperiment(labelled=am_corpus,
                                        document_features_tr='J+N+V',
                                        document_features_ev='J+N+V',
                                        expansions=_make_expansions(vectors=v, k=k))
        e.save(force_insert=True)


@printing_decorator
def cleaned_wiki():
    for percent in [1, 15] + list(range(10, 101, 10)):
        v = vectors_from_settings('cwiki', 'word2vec', 'Add', 100, percent=percent)
        e = db.ClassificationExperiment(labelled=am_corpus, expansions=_make_expansions(vectors=v))
        e.save(force_insert=True)


@printing_decorator
def sentiment_task():
    # with best replacement model
    v = vectors_from_settings('cwiki', 'word2vec', 'Add', 100)

    e = db.ClassificationExperiment(labelled=maas_corpus, expansions=_make_expansions(vectors=v))
    e.save(force_insert=True)

    e = db.ClassificationExperiment(labelled=mr_corpus, expansions=_make_expansions(vectors=v))
    e.save(force_insert=True)

    # with best VQ model
    v = vectors_from_settings('wiki', 'word2vec', 'Add', 100)
    for num_cl in [100, 500, 2000]:
        e = db.ClassificationExperiment(labelled=maas_corpus,
                                        clusters=clusters_from_settings(v, num_cl))
        e.save(force_insert=True)

    for corpus, algo in zip(['cwiki', 'wiki'], ['word2vec', 'glove']):
        v = vectors_from_settings(corpus, algo, 'Add', 100)
        e = db.ClassificationExperiment(labelled=maas_corpus,
                                        clusters=clusters_from_settings(v, 100))
        e.save(force_insert=True)


@printing_decorator
def wiki15_k30():
    """this complements multivectors_higher_k, which is missing two bits baseline to compare against"""
    v = vectors_from_settings('wiki', 'word2vec', 'Add', 100, percent=15)
    e = db.ClassificationExperiment(labelled=am_corpus, expansions=_make_expansions(vectors=v, k=30))
    e.save(force_insert=True)

    v = vectors_from_settings('wiki', 'word2vec', 'Add', 100, percent=15, rep=3, avg=True)
    e = db.ClassificationExperiment(labelled=am_corpus, expansions=_make_expansions(vectors=v, k=30))
    e.save(force_insert=True)


@printing_decorator
def reuters_wiki_gigaw():
    for unlab in 'wiki cwiki gigaw'.split():
        for algo in 'word2vec glove count_windows count_dependencies'.split():
            if algo == 'word2vec' and unlab == 'wiki':
                # did that experiment earlier
                continue
            try:
                v = vectors_from_settings(unlab, algo, 'Add', 100, percent=100)
            except AssertionError:
                # haven't got them vectors, eg glove on cwiki. whatever
                continue
            e = db.ClassificationExperiment(labelled=r2_corpus, expansions=_make_expansions(vectors=v))
            e.save(force_insert=True)


@printing_decorator
def wikipedia_w2v_amazon_with_repeats_k100():
    unlab = 'wiki'
    for rep in [0, 1, 2]:
        for _, algo, composer_name, dims in word2vec_vector_settings():
            v = vectors_from_settings(unlab, algo, composer_name, dims, percent=15, rep=rep)
            e = db.ClassificationExperiment(labelled=am_corpus, expansions=_make_expansions(vectors=v, k=100))
            e.save(force_insert=True)


def write_conf_files():
    check_experiments()

    print('Writing conf files')
    megasuperbase_conf_file = 'conf/exp1-superbase.conf'
    experiments = db.ClassificationExperiment.select()
    for exp in experiments:
        if exp.id % 50 == 0:
            print('Writing exp %d' % exp.id)
        # sanity check
        experiment_dir = 'conf/exp%d' % exp.id
        mkdirs_if_not_exists(experiment_dir)

        conf_file = os.path.join(experiment_dir, 'exp%d.conf' % exp.id)
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
            enable_rand_neigh = exp.expansions.vectors is not None and \
                                exp.expansions.vectors.algorithm == 'random_neigh'
            conf['feature_extraction']['random_neighbour_thesaurus'] = enable_rand_neigh
            if enable_rand_neigh:
                conf['vector_sources']['neighbours_file'] = []
            else:
                neig_f = exp.expansions.vectors.path if exp.expansions.vectors else ''
                conf['vector_sources']['neighbours_file'] = neig_f.split(',')
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
            conf['vector_sources']['neighbours_file'] = []
            conf['feature_extraction']['vectorizer'] = \
                'thesisgenerator.plugins.kmeans_disco.KmeansVectorizer'
            conf['vector_sources']['clusters_file'] = exp.clusters.path
            # the features of the document are cluster ids, not phrases
            # no point in checking in they are in the thesaurus
            conf['feature_selection']['must_be_in_thesaurus'] = False
            # not used, remove
            del conf['feature_extraction']['sim_compressor']

        with open(conf_file, 'wb') as inf:
            conf.write(inf)

        # we've just written a conf file to a directory that may have contained results from before
        # check those results were done with the same configuration we just created. This is needed
        # because as we reorder experiments existing results may end up with a different ID
        # otherwise the experiment will have to be re-run
        # only checking some of the important parameters
        previous_conf_file = 'conf/exp{0}/output/exp{0}.conf'.format(exp.id)
        if os.path.exists(previous_conf_file):
            old_conf, _ = parse_config_file(previous_conf_file, quit_on_error=False)
            for a, b in [(old_conf['vector_sources']['neighbours_file'],
                          conf['vector_sources']['neighbours_file']),
                         (old_conf['feature_extraction']['decode_token_handler'],
                          conf['feature_extraction']['decode_token_handler']),
                         (old_conf['training_data'], conf['training_data'])]:
                if a != b:
                    print('Exp: %d, was %r, is now %r' % (exp.id, a, b))


def check_experiments():
    # verify experiments aren't being duplicated
    experiments = list(db.ClassificationExperiment.select())
    if len(set(experiments)) != len(experiments):
        dupl = [x for x in experiments if x.__hash__() == Counter(experiments).most_common(1)[0][0].__hash__()]
        for x in dupl:
            print(x._key(), x.expansions._key(), x.clusters)
        raise ValueError('Duplicated experiments exist: %s' % dupl)


def write_metafiles():
    """
    Dynamically decide whether to run an experiment. Used when I ran one long experiment and don't want to repeat it,
    but don't want to delete it from the list of experiments either. With this file I can just submit all experiments
    and delete the ones that shouldn't be ran.

    :param experiments:
    """
    with open('slow_experiments.txt', 'w') as outf:
        for e in db.ClassificationExperiment.select():
            try:
                if e.expansions.k >= 250 or e.clusters.num_clusters > 2000 or \
                        (e.expansions.vectors.reorder and e.expansions.vectors.rep > 3):
                    outf.write(str(e.id) + '\n')
            except AttributeError:
                # experiment doesnt have expansions or clusters, it is some baseline
                pass


if __name__ == '__main__':
    prefix = '/lustre/scratch/inf/mmb28/thesisgenerator/sample-data'
    techtc_corpora = sorted(list(os.path.join(*x.split(os.sep)[-2:]) \
                                 for x in glob('%s/techtc100-clean/*' % prefix) \
                                 if not x.endswith('.gz')))
    r2_corpus = 'reuters21578/r8-tagged-grouped'
    mr_corpus = 'movie-reviews-tagged'
    am_corpus = 'amazon_grouped-tagged'
    maas_corpus = 'aclImdb-tagged'
    # havent added maas to all_corpora to avoid changing the ids of long running amazon jobs
    all_corpora = techtc_corpora + [r2_corpus, mr_corpus, am_corpus]

    ################################################################
    # NOUN PHRASES
    ################################################################
    random_baselines(corpora=[am_corpus, r2_corpus])
    nondistributional_baselines(corpora=[am_corpus, r2_corpus])
    nondistributional_baselines(document_features_ev='J+N+AN+NN')

    all_standard_experiments()
    hybrid_experiments_word2vec_gigaw()
    varying_k_with_w2v_on_amazon()
    # wikipedia experiments on amazon
    initial_wikipedia_w2v_amazon_with_repeats()

    # other more recent stuff
    corrupted_w2v_wiki(am_corpus)
    glove_vectors_amazon()
    # only up to 90% to avoid duplicating w2v-gigaw-100% (part of "standard experiments")
    w2v_wiki_learning_curve(percent=[1, 10, 20, 40, 60, 80])
    # show R2 is too small for a meaningful comparison
    w2v_wiki_learning_curve(percent=[1, 10, 20, 30, 50, 40, 60, 70, 80, 90, 100], corpus=r2_corpus)
    corrupted_w2v_wiki(r2_corpus)  # no noise@100% done as part of learning curve
    an_only_nn_only_experiments_amazon()
    equalised_coverage_experiments()
    with_unigrams_at_decode_time()
    with_lexical_overlap()

    ################################################################
    # VERB PHRASES
    ################################################################

    # pad with J+N to increase performance a wee bit
    nondistributional_baselines(document_features_tr='J+N+V+SVO',
                                document_features_ev='SVO')
    verb_phrases_svo()

    ################################################################
    # CLUSTERING
    ################################################################
    kmeans_experiments()

    ################################################################
    # STUFF I DID NOT RUN AT FIRST BUT SHOULD
    ################################################################
    varying_k_with_w2v_on_amazon(ks=[250, 500, 1000])  # push k param further
    equalised_coverage_experiments_v2()

    # various other experiments that aren't as interesting
    # different_neighbour_strategies() # this takes a long time

    kmeans_experiments(min_id=31, max_id=35)
    kmeans_experiments(min_id=31, max_id=35, labelled=r2_corpus)

    multivectors()
    equalised_coverage_experiments_v2(composers=[AdditiveComposer],
                                      percent_reduce_from=[20, 30, 50, 40, 60, 70, 80, 90])

    verb_phrases_svo(composers=[FrobeniusAdd, FrobeniusMult])
    for prt in [1, 10, 20]:
        equalised_coverage_experiments_v2(composers=[AdditiveComposer],
                                          percent_reduce_from=[20, 40, 60, 80, 100],
                                          percent_reduce_to=prt)
    unigram_only_vary_k()
    cleaned_wiki()
    wikipedia_w2v_R2_repeats()
    multivectors_higher_k()
    kmeans_experiments(min_id=36, max_id=47)
    sentiment_task()
    wiki15_k30()
    corrupted_w2v_wiki(r2_corpus, k=30, include_zero=True)
    corrupted_w2v_wiki(r2_corpus, k=60, include_zero=True)
    reuters_wiki_gigaw()
    wikipedia_w2v_amazon_with_repeats_k100()
    # 100% done earlier as a part of noise-corrupted experiments
    w2v_wiki_learning_curve(percent=[1, 10, 20, 30, 50, 40, 60, 70, 80, 90], corpus=r2_corpus, k=30)
    w2v_wiki_learning_curve(percent=[1, 10, 20, 30, 50, 40, 60, 70, 80, 90], corpus=r2_corpus, k=60)

    # # maas IMDB sentiment experiments
    # baselines(corpora=[maas_corpus])
    # random_vectors(maas_corpus)
    all_standard_experiments(labelled=[maas_corpus])
    equalised_coverage_experiments_v3()

    print('Total experiments: %d' % len(list(db.ClassificationExperiment.select())))
    write_conf_files()
    write_metafiles()
