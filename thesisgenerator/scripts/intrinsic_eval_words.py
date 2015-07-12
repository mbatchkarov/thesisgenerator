import argparse
import logging
import os
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from discoutils.thesaurus_loader import Vectors as vv
import pandas as pd
import numpy as np


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


def datasets():
    yield 'ws353', _ws353()
    yield 'mc', _mc()
    yield 'rg', _rg()
    yield 'men', _men()


def _intrinsic_eval(vectors, intrinsic_dataset, noise=0, reload=True):
    v = vv.from_tsv(vectors, noise=noise) if reload else vectors

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

    # bootstrap a CI for the data
    for boot_i in range(500):
        if boot_i % 100 == 0:
            logging.info('Doing boostrap number %d', boot_i)
        model_sims, human_sims = [], []
        missing = 0
        idx = np.random.randint(0, len(intrinsic_dataset), len(intrinsic_dataset))
        for w1, w2, human in zip(intrinsic_dataset.w1[idx],
                                 intrinsic_dataset.w2[idx],
                                 intrinsic_dataset.sim[idx]):
            v1, v2 = get_vector_for(w1), get_vector_for(w2)
            if v1 and v2:
                model_sims.append(cosine_similarity(v1[0], v2[0])[0][0])
                human_sims.append(human)
            else:
                missing += 1

        relaxed, rel_pval = spearmanr(model_sims, human_sims)

        model_sims += [0] * missing
        human_sims += [0] * missing
        strict, str_pval = spearmanr(model_sims, human_sims)

        yield strict, relaxed, noise, rel_pval, str_pval, missing / len(intrinsic_dataset)


def noise_eval():
    """
    Test: intrinsic eval on noise-corrupted vectors

    Add noise as usual, evaluated intrinsically.
    """
    paths = ['../FeatureExtractionToolkit/word2vec_vectors/word2vec-gigaw-100perc.unigr.strings.rep0',
             '../FeatureExtractionToolkit/word2vec_vectors/word2vec-wiki-15perc.unigr.strings.rep0']
    names = ['w2v-giga-100',
             'w2v-wiki-15']
    noise_data = []
    for dname, df in datasets():
        for vname, path in zip(names, paths):
            logging.info('starting %s %s', dname, vname)
            for noise in np.arange(0, 3.1, .2):
                for strict, relaxed, noise, rel_pval, str_pval, _ in _intrinsic_eval(path, df, noise):
                    noise_data.append((vname, dname, noise, 'strict', strict, str_pval))
                    noise_data.append((vname, dname, noise, 'relaxed', relaxed, rel_pval))
    noise_df = pd.DataFrame(noise_data, columns=['vect', 'test', 'noise', 'kind', 'corr', 'pval'])
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
        vectors = vv.from_tsv(os.path.join(prefix, filename))
        for dname, df in datasets():
            for strict, relaxed, noise, rel_pval, str_pval, missing in _intrinsic_eval(vectors, df, 0, reload=False):
                curve_data.append((percent, dname, missing) + ('strict', strict, str_pval))
                curve_data.append((percent, dname, missing) + ('relaxed', relaxed, rel_pval))

    curve_df = pd.DataFrame(curve_data, columns=['percent', 'test', 'missing', 'kind', 'corr', 'pval'])
    curve_df.to_csv('intrinsic_learning_curve_word_level.csv')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', choices=('noise', 'curve'), required=True)
    parameters = parser.parse_args()

    if parameters.stages == 'noise':
        noise_eval()
    if parameters.stages == 'curve':
        learning_curve_wiki()
