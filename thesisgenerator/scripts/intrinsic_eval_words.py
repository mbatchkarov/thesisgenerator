from collections import defaultdict
from itertools import chain
import os, sys
from sklearn.metrics import accuracy_score

sys.path.append('.')
from thesisgenerator.composers.vectorstore import (AdditiveComposer,
                                                   RightmostWordComposer,
                                                   MultiplicativeComposer)
from discoutils.thesaurus_loader import Vectors
from joblib.parallel import delayed, Parallel
import argparse
import logging
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import pandas as pd
import numpy as np

ALLOW_OVERLAP = False
PATHS = ['../FeatureExtractionToolkit/word2vec_vectors/word2vec-gigaw-100perc.unigr.strings.rep0',
         '../FeatureExtractionToolkit/word2vec_vectors/word2vec-wiki-15perc.unigr.strings.rep0']
NAMES = ['w2v-giga-100', 'w2v-wiki-15']
NBOOT = 100

def _ws353():
    return pd.read_csv('similarity-data/wordsim353/combined.csv',
                       names=['w1', 'w2', 'sim'])


def _mc():
    return pd.read_csv('similarity-data/miller-charles.txt',
                       names=['w1', 'w2', 'sim'], sep='\t')


def _rg():
    return pd.read_csv('similarity-data/rub-gooden.txt',
                       names=['w1', 'w2', 'sim'], sep='\t')


def _men():
    df = pd.read_csv('similarity-data/MEN/MEN_dataset_lemma_form_full',
                     names=['w1', 'w2', 'sim'], sep=' ')

    def _convert_tag(word):
        return '%s/%s' % (word[:-2], word[-1].upper())

    df.w1 = df.w1.map(_convert_tag)
    df.w2 = df.w2.map(_convert_tag)
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

    def get_vector_for(word):
        vectors = []
        if word[-2] == '/':
            # pos tag is there already, let's just do it
            if word in v:
                vectors.append(v.get_vector(word))
        else:
            # what could the pos tag be?
            for pos in 'JNV':
                candidate = '%s/%s' % (word.lower(), pos)
                if candidate in v:
                    vectors.append(v.get_vector(candidate))
        if len(vectors) > 1:
            pass
        # logging.info('multiple vectors for', word, len(vectors))
        return vectors

    model_sims, human_sims = [], []
    missing = 0
    for w1, w2, human in zip(intrinsic_dataset.w1,
                             intrinsic_dataset.w2,
                             intrinsic_dataset.sim):
        v1, v2 = get_vector_for(w1), get_vector_for(w2)
        if v1 and v2:
            model_sims.append(cosine_similarity(v1[0], v2[0])[0][0])
            human_sims.append(human)
        else:
            missing += 1

    # bootstrap a CI for the data
    res = []
    for boot_i in range(NBOOT):
        idx = np.random.randint(0, len(model_sims), len(model_sims))
        relaxed, rel_pval = spearmanr(np.array(model_sims)[idx],
                                      np.array(human_sims)[idx])

        model_sims += [0] * missing
        human_sims += [0] * missing
        strict, str_pval = spearmanr(np.array(model_sims)[idx],
                                     np.array(human_sims)[idx])

        res.append([strict, relaxed, noise, rel_pval, str_pval, missing / len(intrinsic_dataset), boot_i])
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
    noise_df = pd.DataFrame(noise_data, columns=['vect', 'test', 'noise', 'kind', 'corr', 'pval', 'folds'])
    noise_df.to_csv('intrinsic_noise_word_level.csv')


def learning_curve_wiki():
    """
    Test: Learning curve

    Evaluate vectors intrinsically as more unlabelled training data is added
    """
    prefix = '/mnt/lustre/scratch/inf/mmb28/FeatureExtractionToolkit/word2vec_vectors/composed'
    paths = [(1, 'AN_NN_word2vec-wiki_1percent-rep0_Add.events.filtered.strings'),
             (10, 'AN_NN_word2vec-wiki_10percent-rep0_Add.events.filtered.strings'),
             (15, 'AN_NN_word2vec-wiki_15percent-rep0_Add.events.filtered.strings'),
             (20, 'AN_NN_word2vec-wiki_20percent-rep0_Add.events.filtered.strings'),
             (30, 'AN_NN_word2vec-wiki_30percent-rep0_Add.events.filtered.strings'),
             (40, 'AN_NN_word2vec-wiki_40percent-rep0_Add.events.filtered.strings'),
             (50, 'AN_NN_word2vec-wiki_50percent-rep0_Add.events.filtered.strings'),
             (60, 'AN_NN_word2vec-wiki_60percent-rep0_Add.events.filtered.strings'),
             (70, 'AN_NN_word2vec-wiki_70percent-rep0_Add.events.filtered.strings'),
             (80, 'AN_NN_word2vec-wiki_80percent-rep0_Add.events.filtered.strings'),
             (90, 'AN_NN_word2vec-wiki_90percent-rep0_Add.events.filtered.strings'),
             (100, 'AN_NN_word2vec-wiki_100percent-rep0_Add.events.filtered.strings')
             ]
    curve_data = []
    for percent, filename in paths:
        logging.info('Doing percentage: %d', percent)
        vectors = Vectors.from_tsv(os.path.join(prefix, filename))
        for dname, intr_data in word_level_datasets():
            for strict, relaxed, noise, rel_pval, str_pval, missing, boot_i in _intrinsic_eval_words(vectors,
                                                                                                     intr_data,
                                                                                                     0,
                                                                                                     reload=False):
                curve_data.append((percent, dname, missing, 'strict', strict, str_pval, boot_i))
                curve_data.append((percent, dname, missing, 'relaxed', relaxed, rel_pval, boot_i))
    curve_df = pd.DataFrame(curve_data, columns=['percent', 'test', 'missing', 'kind', 'corr', 'pval', 'folds'])
    curve_df.to_csv('intrinsic_learning_curve_word_level.csv')


def turney_predict(phrase, possible_answers, composer, unigram_source):
    def _maxint():
        return 1e19

    def _add_tags(phrase):
        words = phrase.split()
        for pos in 'JN':
            yield '{1}/{0}_{2}/N'.format(pos, *words)

    def _add_pos(word):
        for pos in 'NJ':
            yield '{}/{}'.format(word, pos)

    sims = defaultdict(_maxint)
    for candidate_phrase in _add_tags(phrase):
        if candidate_phrase in composer and composer.get_vector(candidate_phrase) is not None:
            phrase_vector = composer.get_vector(candidate_phrase).A
            for wordid, word in enumerate(possible_answers):
                for candidate_word in _add_pos(word):
                    if candidate_word in unigram_source:
                        word_vector = unigram_source.get_vector(candidate_word).A
                        distance = euclidean(phrase_vector.ravel(), word_vector.ravel())
                        if distance < sims[word]:
                            sims[word] = distance
                if wordid == 0 and sims[word] > 1e10:
                    # don't have a word vector for the gold std neighbour
                    return None, None
    if not sims:
        #         print('cant process', phrase)
        return None, None
    else:
        return min(sims, key=sims.get), sims


def turney_measure_accuracy(path, composer_class, df):
    unigram_source = Vectors.from_tsv(path)
    composer = composer_class(unigram_source)

    res = []
    predictions, gold = [], []
    for phrase, candidates in df.iterrows():
        most_similar, _ = turney_predict(phrase, candidates, composer, unigram_source)
        if most_similar:
            predictions.append(most_similar)
            gold.append(candidates[0])

    coverage = len(predictions) / len(df)
    for boot_i in range(NBOOT):
        idx = np.random.randint(0, len(gold), len(gold))
        accuracy = accuracy_score(np.array(predictions)[idx],
                                  np.array(gold)[idx])
        res.append([coverage, accuracy, composer.name, boot_i])
    return res


def turney_evaluation():
    df = _turney2010(ALLOW_OVERLAP)
    if ALLOW_OVERLAP:
        assert list(df.values[0, :]) == ['binary', 'double', 'star', 'dual', 'lumen', 'neutralism', 'keratoplasty']
    else:
        assert list(df.values[0, :]) == ['binary', 'dual', 'lumen', 'neutralism', 'keratoplasty']

    composers = [AdditiveComposer, MultiplicativeComposer, RightmostWordComposer]
    # right/left should always score 0 with overlap

    results = []
    for path, vname in zip(PATHS, NAMES):
        logging.info('Turney test doing %s', vname)
        res = Parallel(n_jobs=4)(delayed(turney_measure_accuracy)(path, comp, df) \
                                 for comp in composers)
        for cov, acc, comp_name, boot in chain.from_iterable(res):
            results.append((vname, comp_name, cov, acc, boot))

    df_res = pd.DataFrame(results, columns=['unigrams', 'composer', 'coverage', 'accuracy', 'folds'])
    df_res.to_csv('intrinsic_turney_phraselevel.csv')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', choices=('noise', 'curve', 'turney'), required=True)
    parameters = parser.parse_args()

    if parameters.stages == 'noise':
        noise_eval()
    if parameters.stages == 'curve':
        learning_curve_wiki()
    if parameters.stages == 'turney':
        turney_evaluation()

    logging.info('Done')
