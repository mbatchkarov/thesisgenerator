import logging
import os
from itertools import groupby
import numpy as np
import pandas as pd
from discoutils.thesaurus_loader import Vectors
from discoutils.tokens import DocumentFeature, itemgetter
from discoutils.misc import mkdirs_if_not_exists
from thesisgenerator.composers.vectorstore import CopyObject, compose_and_write_vectors


VERBS_HDF_DIR = '/mnt/lustre/scratch/inf/mmb28/FeatureExtractionToolkit/categorical/'
MIN_SVO_PER_VERB = 3  # todo does this filter exist in the original paper?


def train_verb_tensors(svos_file, noun_vectors_file, output_filename):
    """
    Trains Verb-bar matrices, as described in Milajevs et al (EMNLP-14, §3)
    :param svos_file: file containing a list of all SVOs in unlabelled data, one per line. May contain other document
     features too. Such a file is output by `find_all_NPs.py`, which is called from `observed_vectors.py`
    :param noun_vectors_file: a vector store containing noun vectors
    :param output_filename: name of output file- must identify the noun vectors and the unlabelled corpus
    """
    mkdirs_if_not_exists(os.path.dirname(output_filename))

    v = Vectors.from_tsv(noun_vectors_file)

    with open(svos_file) as infile:
        phrases = set()
        for line in infile:
            if DocumentFeature.from_string(line.strip()).type == 'SVO':
                phrases.add(tuple(line.strip().split('_')))
    phrases = [(subj, verb, obj) for subj, verb, obj in phrases if subj in v and obj in v]
    phrases = sorted(phrases, key=itemgetter(1))
    logging.info('Found %d SVOs in list', len(phrases))

    verb_tensors = dict()
    for verb, svos in groupby(phrases, itemgetter(1)):
        svos = list(svos)
        if len(svos) < MIN_SVO_PER_VERB:
            continue
        logging.info('Training matrix for %s from %d SVOs', verb, len(svos))
        vt = np.sum(np.outer(v.get_vector(subj).A, v.get_vector(obj).A) for subj, _, obj in svos)
        verb_tensors[verb] = vt

    logging.info('Trained %d verb matrices, saving...', len(verb_tensors))
    for verb, tensor in verb_tensors.items():
        df = pd.DataFrame(tensor)
        df.to_hdf(output_filename, verb.split('/')[0], complevel=9, complib='zlib')


def _nouns_only_filter(s, dfs):
    return dfs.type == '1-GRAM' and dfs.tokens[0].pos == 'N'


def compose_categorical(unigram_vectors, verb_tensors_filename):
    """
    :param unigram_vectors: a vector store containing noun vectors
    :param verb_tensors_filename: filename of output of training stage
    """

    SVD_DIMS = 100  # todo urgh, this first half is awful. whatever
    corpus, features = 10, 13  # todo change corpus here
    dataset_name = 'gigaw' if corpus == 10 else 'wiki'  # short name of input corpus
    features_name = 'wins' if features == 13 else 'deps'  # short name of input corpus
    outdir = '/mnt/lustre/scratch/inf/mmb28/FeatureExtractionToolkit/exp%d-%d-composed-ngrams' % (corpus, features)

    compose_and_write_vectors(unigram_vectors,
                              '%s-%s' % (dataset_name, SVD_DIMS),
                              [CopyObject],  # todo can add other categorical models here easily
                              categorical_vector_matrix_file=verb_tensors_filename,
                              output_dir=outdir, dense_hd5=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %(message)s")

    noun_path = '/mnt/lustre/scratch/inf/mmb28/FeatureExtractionToolkit/exp10-13b/' \
                'exp10-with-obs-phrases-SVD100.events.filtered.strings'
    svos_path = '/mnt/lustre/scratch/inf/mmb28/DiscoUtils/gigaw_NPs_in_MR_R2_TechTC_am_maas.uniq.30.txt'

    trained_verb_matrices_file = os.path.join(VERBS_HDF_DIR, 'gigaw-wins-vector-matrices.hdf')
    train_verb_tensors(svos_path, noun_path, trained_verb_matrices_file)
    compose_categorical(noun_path, trained_verb_matrices_file)
