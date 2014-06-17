from itertools import chain
import logging
import os
from pprint import pprint
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from joblib import Parallel, delayed
from numpy import hstack
from sklearn.pipeline import Pipeline
from thesisgenerator.composers.feature_selectors import VectorBackedSelectKBest, MetadataStripper
from thesisgenerator.composers.vectorstore import *
from thesisgenerator.plugins.bov import ThesaurusVectorizer
from thesisgenerator.utils.data_utils import load_tokenizer, tokenize_data, load_text_data_into_memory


def _loop_body(composer_class, output_dir, pipeline, pretrained_Baroni_composer_file, short_vector_dataset_name,
               unigram_source, x_ev, x_tr, y_ev, y_tr):
    # whatever files we need to (entries index, as composition creates new entries)
    if composer_class == BaroniComposer:
        composers = [BaroniComposer(unigram_source, pretrained_Baroni_composer_file)]
    else:
        composers = [composer_class(unigram_source)]

    # the next 2 lines ensure 1-GRAMs are contained in the composer and returned as neighbours of phrases
    s1 = UnigramDummyComposer(unigram_source)
    composers.append(s1)

    vector_source = CompositeVectorSource(composers, sim_threshold=0, include_self=False)
    fit_args = {
        'stripper__vector_source': vector_source,
        'vect__vector_source': vector_source,
        'fs__vector_source': vector_source,
    }
    _ = pipeline.fit_transform(x_tr + x_ev, y=hstack([y_tr, y_ev]), **fit_args)

    output_files = ('AN_NN_%s_%s.events.filtered.strings' % (short_vector_dataset_name, composers[0].name),
                    'AN_NN_%s_%s.entries.filtered.strings' % (short_vector_dataset_name, composers[0].name),
                    'AN_NN_%s_%s.features.filtered.strings' % (short_vector_dataset_name, composers[0].name))
    output_files = [os.path.join(output_dir, x) for x in output_files]

    pipeline.steps[2][1].vector_source.write_vectors_to_disk({'AN', 'NN', '1-GRAM'}, *output_files)


def compose_and_write_vectors(unigram_vector_paths, short_vector_dataset_name, classification_data_paths,
                              pretrained_Baroni_composer_file,
                              output_dir='.', composer_classes='bar'):
    """
    Extracts all composable features from a labelled classification corpus and dumps a composed vector for each of them
    to disk
    :param unigram_vector_paths: a list of files in Byblo events format that contain vectors for all unigrams. This
    will be used in the composition process
    :param classification_data_paths: Corpora to extract features from. Type: [ [train1, test1], [train2, test2] ]
    :param pretrained_Baroni_composer_file: path to pre-trained Baroni AN/NN composer file
    :param output_dir:
    :param composer_classes: what composers to use
    :return:
    :rtype: list of strings
    """
    tokenized_data = ([], [], [], [])
    for train, test in classification_data_paths:
        raw_data, data_ids = load_text_data_into_memory(
            training_path=train,
            test_path=test,
            shuffle_targets=False
        )

        tokenizer = load_tokenizer(
            joblib_caching=False,
            normalise_entities=False,
            use_pos=True,
            coarse_pos=True,
            lemmatize=True,
            lowercase=True,
            remove_stopwords=True,
            remove_short_words=False)
        data_in_this_dir = tokenize_data(raw_data, tokenizer, data_ids)
        for x, y in zip(tokenized_data, data_in_this_dir):
            x.extend(y)

    pipeline = Pipeline([
        ('vect', ThesaurusVectorizer(ngram_range=(1, 1), min_df=1, use_tfidf=False,
                                     extract_VO_features=False, extract_SVO_features=False,
                                     unigram_feature_pos_tags=['N', 'J'])),
        ('fs', VectorBackedSelectKBest(must_be_in_thesaurus=True)),
        ('stripper', MetadataStripper(nn_algorithm='brute', build_tree=False))
    ])

    x_tr, y_tr, x_ev, y_ev = tokenized_data
    unigram_source = UnigramVectorSource(unigram_vector_paths, reduce_dimensionality=False)
    Parallel(n_jobs=1)(delayed(_loop_body)(composer_class,
                                           output_dir,
                                           pipeline,
                                           pretrained_Baroni_composer_file,
                                           short_vector_dataset_name,
                                           unigram_source,
                                           x_ev, x_tr,
                                           y_ev, y_tr) for composer_class in composer_classes)


n_jobs = 4
classification_data_path = (
    '/mnt/lustre/scratch/inf/mmb28/thesisgenerator/sample-data/reuters21578/r8train-tagged-grouped',
    '/mnt/lustre/scratch/inf/mmb28/thesisgenerator/sample-data/reuters21578/r8test-tagged-grouped')

classification_data_path_mr = (
    '/mnt/lustre/scratch/inf/mmb28/thesisgenerator/sample-data/movie-reviews-train-tagged',
    '/mnt/lustre/scratch/inf/mmb28/thesisgenerator/sample-data/movie-reviews-test-tagged')


