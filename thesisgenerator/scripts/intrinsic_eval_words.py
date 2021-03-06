from collections import defaultdict, Counter
from itertools import chain
import os, sys
from sklearn.metrics import accuracy_score

sys.path.append('.')
from thesisgenerator.composers.vectorstore import (AdditiveComposer,
                                                   RightmostWordComposer,
                                                   MultiplicativeComposer,
                                                   AverageComposer)
from thesisgenerator.plugins.multivectors import MultiVectors
from discoutils.thesaurus_loader import Vectors
from joblib.parallel import delayed, Parallel
import argparse
import logging
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import pandas as pd
import numpy as np

ALLOW_OVERLAP = True
PATHS = ['../FeatureExtractionToolkit/word2vec_vectors/word2vec-gigaw-nopos-100perc.unigr.strings.rep0',
         '../FeatureExtractionToolkit/word2vec_vectors/word2vec-wiki-nopos-15perc.unigr.strings.rep0']
PATHS2 = PATHS + ['../FeatureExtractionToolkit/word2vec_vectors/word2vec-wiki-nopos-100perc.unigr.strings.rep0']
NAMES = ['w2v-giga-100', 'w2v-wiki-15']
NAMES2 = NAMES + ['w2v-wiki-100']
NBOOT = 500


def _ws353():
    df = pd.read_csv('similarity-data/wordsim353/combined.csv',
                     names=['w1', 'w2', 'sim'])
    df.w1 = df.w1.map(str.lower)
    df.w2 = df.w2.map(str.lower)
    return df


def _mc():
    return pd.read_csv('similarity-data/miller-charles.txt',
                       names=['w1', 'w2', 'sim'], sep='\t')


def _rg():
    return pd.read_csv('similarity-data/rub-gooden.txt',
                       names=['w1', 'w2', 'sim'], sep='\t')


def _men():
    df = pd.read_csv('similarity-data/MEN/MEN_dataset_lemma_form_full',
                     names=['w1', 'w2', 'sim'], sep=' ')

    def _remove_pos_tag(word):
        return word[:-2]

    df.w1 = df.w1.map(_remove_pos_tag)
    df.w2 = df.w2.map(_remove_pos_tag)
    return df


def _turney2010(allow_overlap=True):
    df = pd.read_csv('similarity-data/turney-2012-jair-phrasal.txt',
                     names=['phrase'] + ['word%d' % i for i in range(1, 7)],
                     comment='#', sep='|', index_col=0)
    for col in df.columns:
        df[col] = list(map(str.strip, df[col].values))
    df.index = list(map(str.strip, df.index))
    return df if allow_overlap else df.drop(['word1', 'word2'], axis=1)


def word_level_datasets():
    yield 'ws353', _ws353()
    yield 'mc', _mc()
    yield 'rg', _rg()
    yield 'men', _men()


def _intrinsic_eval_words(vectors, intrinsic_dataset, noise=0, reload=True):
    v = Vectors.from_tsv(vectors, noise=noise) if reload else vectors
    model_sims, human_sims = [], []
    missing = 0
    for w1, w2, human in zip(intrinsic_dataset.w1,
                             intrinsic_dataset.w2,
                             intrinsic_dataset.sim):
        v1, v2 = v.get_vector(w1), v.get_vector(w2)
        if v1 is not None and v2 is not None:
            model_sims.append(cosine_similarity(v1, v2)[0][0])
            human_sims.append(human)
        else:
            missing += 1
    # where model failed to answer insert something at random
    model_sims_w_zeros = model_sims + list(np.random.uniform(min(model_sims), max(model_sims), missing))
    human_sims_w_zeros = human_sims + list(np.random.uniform(min(human_sims), max(human_sims), missing))

    # sanity check: strict accuracy results must be lower than relaxed
    # some allowance for change- random guesses may slightly improve results
    relaxed, _ = spearmanr(model_sims, human_sims)
    strict, _ = spearmanr(model_sims_w_zeros, human_sims_w_zeros)
    if strict >= relaxed:
        logging.warning('Strict correlation higher than relaxed: %.2f, %,2f', strict, relaxed)

    # bootstrap model_sims_w_zeros CI for the data
    res = []
    for boot_i in range(NBOOT):
        idx = np.random.randint(0, len(model_sims), len(model_sims))
        relaxed, rel_pval = spearmanr(np.array(model_sims)[idx],
                                      np.array(human_sims)[idx])

        idx = np.random.randint(0, len(model_sims_w_zeros), len(model_sims_w_zeros))
        strict, str_pval = spearmanr(np.array(model_sims_w_zeros)[idx],
                                     np.array(human_sims_w_zeros)[idx])

        res.append([strict, relaxed, noise, rel_pval, str_pval,
                    missing / len(intrinsic_dataset), boot_i])
    return res


def noise_eval():
    """
    Test: intrinsic eval on noise-corrupted vectors

    Add noise as usual, evaluated intrinsically.
    """
    noise_data = []
    for dname, df in word_level_datasets():
        for vname, path in zip(NAMES, PATHS):
            logging.info('starting %s %s', dname, vname)
            res = Parallel(n_jobs=4)(delayed(_intrinsic_eval_words)(path, df, noise) \
                                     for noise in np.arange(0, 3.1, .2))
            for strict, relaxed, noise, rel_pval, str_pval, _, boot_i in chain.from_iterable(res):
                noise_data.append((vname, dname, noise, 'strict', strict, str_pval, boot_i))
                noise_data.append((vname, dname, noise, 'relaxed', relaxed, rel_pval, boot_i))
    noise_df = pd.DataFrame(noise_data,
                            columns=['vect', 'test', 'noise', 'kind', 'corr', 'pval', 'folds'])
    noise_df.to_csv('intrinsic_noise_word_level.csv')


def learning_curve_wiki():
    """
    Test: Learning curve

    Evaluate vectors intrinsically as more unlabelled training data is added
    """
    prefix = '/lustre/scratch/inf/mmb28/FeatureExtractionToolkit/word2vec_vectors'
    paths = [(1, 'word2vec-wiki-nopos-1perc.unigr.strings.rep0'),
             (10, 'word2vec-wiki-nopos-10perc.unigr.strings.rep0'),
             (15, 'word2vec-wiki-nopos-15perc.unigr.strings.rep0'),
             (20, 'word2vec-wiki-nopos-20perc.unigr.strings.rep0'),
             (30, 'word2vec-wiki-nopos-30perc.unigr.strings.rep0'),
             (40, 'word2vec-wiki-nopos-40perc.unigr.strings.rep0'),
             (50, 'word2vec-wiki-nopos-50perc.unigr.strings.rep0'),
             (60, 'word2vec-wiki-nopos-60perc.unigr.strings.rep0'),
             (70, 'word2vec-wiki-nopos-70perc.unigr.strings.rep0'),
             (80, 'word2vec-wiki-nopos-80perc.unigr.strings.rep0'),
             (90, 'word2vec-wiki-nopos-90perc.unigr.strings.rep0'),
             (100, 'word2vec-wiki-nopos-100perc.unigr.strings.rep0')
             ]
    curve_data = []
    for percent, filename in paths:
        logging.info('Doing percentage: %d', percent)
        vectors = Vectors.from_tsv(os.path.join(prefix, filename))
        for dname, intr_data in word_level_datasets():
            for strict, relaxed, noise, rel_pval, str_pval, missing, boot_i in \
                    _intrinsic_eval_words(vectors, intr_data, 0, reload=False):
                curve_data.append((percent, dname, missing, 'strict', strict, str_pval, boot_i))
                curve_data.append((percent, dname, missing, 'relaxed', relaxed, rel_pval, boot_i))
    curve_df = pd.DataFrame(curve_data,
                            columns=['percent', 'test', 'missing', 'kind', 'corr', 'pval',
                                     'folds'])
    curve_df.to_csv('intrinsic_learning_curve_word_level.csv')


def repeated_runs_w2v():
    prefix = '/lustre/scratch/inf/mmb28/FeatureExtractionToolkit/word2vec_vectors/'
    pattern = os.path.join(prefix, 'word2vec-wiki-nopos-15perc.unigr.strings.rep%d')
    rep_vectors = [Vectors.from_tsv(pattern % i) for i in [0, 1, 2]]
    mv = [MultiVectors(tuple(rep_vectors))]

    data = []
    for v, rep_id in zip(rep_vectors + mv, [0, 1, 2, -2]):
        for dname, intr_data in word_level_datasets():
            for strict, relaxed, noise, rel_pval, str_pval, missing, boot_i in \
                    _intrinsic_eval_words(v, intr_data, 0, reload=False):
                data.append((rep_id, dname, missing, 'strict', strict, str_pval, boot_i))
                data.append((rep_id, dname, missing, 'relaxed', relaxed, rel_pval, boot_i))
    df = pd.DataFrame(data, columns=['rep_id', 'test', 'missing', 'kind', 'corr', 'pval', 'folds'])
    df.to_csv('intrinsic_w2v_repeats_word_level.csv')


def turney_predict(phrase, possible_answers, composer, unigram_source):
    distances = np.nan * np.ones(len(possible_answers))
    phrase = phrase.replace(' ', '_')

    if phrase in composer and composer.get_vector(phrase) is not None:
        phrase_vector = composer.get_vector(phrase).A
        for wordid, word in enumerate(possible_answers):
            if word in unigram_source:
                word_vector = unigram_source.get_vector(word).A
                distances[wordid] = euclidean(phrase_vector.ravel(), word_vector.ravel())

    # strict means we must always predict a neighbour, even if we don't know
    if np.all(np.isnan(distances)):
        # we have absolutely no idea, ignore the question
        relaxed_pred = strict_pred = None
        idx_of_pred = -1  # of strict version
    else:
        idx_of_pred = np.nanargmin(distances)
        relaxed_pred = strict_pred = possible_answers[idx_of_pred]

    if np.isnan(distances[0]):
        # we don't have a word vector for the gold std neighbour, refuse to predict
        # because there's no way to get it right. that is sort of cheating though
        relaxed_pred = None

    return strict_pred, relaxed_pred, np.log10(distances), idx_of_pred


def turney_measure_accuracy(path, composer_class, df):
    unigram_source = Vectors.from_tsv(path)
    composer = composer_class(unigram_source)

    res = []
    str_predictions, str_gold, rel_predictions, rel_gold = [], [], [], []
    indices_of_predictions = []
    for phrase, candidates in df.iterrows():
        strict_pred, relaxed_pred, d, ind = turney_predict(phrase, candidates.values,
                                                           composer, unigram_source)
        indices_of_predictions.append(ind)
        # print(phrase, '--->', strict_pred, relaxed_pred)
        if relaxed_pred:
            rel_predictions.append(relaxed_pred)
            rel_gold.append(candidates[0])
        if strict_pred:
            str_predictions.append(strict_pred)
            str_gold.append(candidates[0])
    print(path, composer_class.name, Counter(indices_of_predictions))
    str_coverage = len(str_predictions) / len(df)
    rel_coverage = len(rel_predictions) / len(df)
    if len(str_gold) == 0 or len(rel_gold) == 0:
        print("WARNING: absolutely zero coverage, cannot bootstrap. Joblib may propagate a non-existant syntax error")
    for boot_i in range(NBOOT):
        idx = np.random.randint(0, len(str_gold), len(str_gold))
        str_accuracy = accuracy_score(np.array(str_predictions)[idx],
                                      np.array(str_gold)[idx])

        idx = np.random.randint(0, len(rel_gold), len(rel_gold))
        rel_accuracy = accuracy_score(np.array(rel_predictions)[idx],
                                      np.array(rel_gold)[idx])
        res.append([str_coverage, rel_coverage, str_accuracy, rel_accuracy, composer.name, boot_i])
    return res


def turney_evaluation():
    df = _turney2010(ALLOW_OVERLAP)
    if ALLOW_OVERLAP:
        assert list(df.values[0, :]) == ['binary', 'double', 'star', 'dual', 'lumen', 'neutralism',
                                         'keratoplasty']
    else:
        assert list(df.values[0, :]) == ['binary', 'dual', 'lumen', 'neutralism', 'keratoplasty']

    composers = [AdditiveComposer, MultiplicativeComposer, RightmostWordComposer, AverageComposer]
    # right/left should always score 0 with overlap

    results = []
    for path, vname in zip(PATHS2, NAMES2):
        logging.info('Turney test doing %s', vname)
        res = Parallel(n_jobs=4)(delayed(turney_measure_accuracy)(path, comp, df) \
                                 for comp in composers)
        for str_cov, rel_cov, str_acc, rel_acc, comp_name, boot in chain.from_iterable(res):
            results.append((vname, comp_name, str_cov, 'strict', str_acc, boot))
            results.append((vname, comp_name, rel_cov, 'relaxed', rel_acc, boot))

    df_res = pd.DataFrame(results,
                          columns=['unigrams', 'composer', 'coverage', 'kind', 'accuracy', 'folds'])
    df_res.to_csv('intrinsic_turney_phraselevel.csv')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%("
                               "levelname)s : %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument('--stages',
                        choices=('noise', 'curve', 'turney', 'repeats'),
                        required=True)
    parameters = parser.parse_args()

    if parameters.stages == 'noise':
        noise_eval()
    if parameters.stages == 'curve':
        learning_curve_wiki()
    if parameters.stages == 'turney':
        turney_evaluation()
    if parameters.stages == 'repeats':
        repeated_runs_w2v()
    logging.info('Done')
