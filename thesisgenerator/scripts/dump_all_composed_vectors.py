from glob import glob
from itertools import chain
import logging
import os
from pprint import pprint
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from numpy import hstack
from sklearn.pipeline import Pipeline
from thesisgenerator.composers.feature_selectors import VectorBackedSelectKBest
from thesisgenerator.composers.vectorstore import *
from thesisgenerator.plugins.bov import ThesaurusVectorizer
from thesisgenerator.utils.data_utils import load_tokenizer, load_text_data_into_memory, tokenize_data
from discoutils.io_utils import write_vectors_to_disk


def compose_and_write_vectors(unigram_vectors_path, short_vector_dataset_name, classification_data_paths,
                              pretrained_Baroni_composer_file,
                              output_dir='.', composer_classes='bar'):
    """
    Extracts all composable features from a labelled classification corpus and dumps a composed vector for each of them
    to disk
    :param unigram_vectors_path: a file in Byblo events format that contain vectors for all unigrams. This
    will be used in the composition process
    :param classification_data_paths: Corpora to extract features from. Type: [ [train1, test1], [train2, test2] ]
    :param pretrained_Baroni_composer_file: path to pre-trained Baroni AN/NN composer file
    :param output_dir:
    :param composer_classes: what composers to use
    :return:
    :rtype: list of strings
    """
    tokenized_data = ([], [], [], [])
    for dataset in classification_data_paths:

        raw_data, data_ids = load_text_data_into_memory(dataset)
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
            if y is not None:
                x.extend(y)

    pipeline = Pipeline([
        ('vect', ThesaurusVectorizer(ngram_range=(0, 0), min_df=5, use_tfidf=False, # only compose frequent phrases
                                     extract_VO_features=False, extract_SVO_features=False,
                                     unigram_feature_pos_tags=['N', 'J'])),
        ('fs', VectorBackedSelectKBest(must_be_in_thesaurus=True)),
    ])

    x_tr, y_tr, x_ev, y_ev = tokenized_data
    vectors = Vectors.from_tsv(unigram_vectors_path,
                               # ensure there's only unigrams in the set of unigram vectors
                               row_filter=lambda x, y: y.tokens[0].pos in {'N', 'J'} and y.type == '1-GRAM')

    # doing this loop in parallel isn't worth it as pickling or shelving `vectors` is so slow
    # it negates any gains from using multiple cores
    for composer_class in composer_classes:
        if composer_class == BaroniComposer:
            composer = BaroniComposer(vectors, pretrained_Baroni_composer_file)
        else:
            composer = composer_class(vectors)

        fit_args = {
            'vect__vector_source': composer,
            'fs__vector_source': composer,
        }
        _, vocabulary = pipeline.fit_transform(x_tr + x_ev, y=hstack([y_tr, y_ev]), **fit_args)
        mat, cols, rows = composer.compose_all(vocabulary.keys())

        events_path = os.path.join(output_dir,
                                   'AN_NN_%s_%s.events.filtered.strings' % (short_vector_dataset_name, composer.name))
        entries_path = os.path.join(output_dir,
                                    'AN_NN_%s_%s.entries.filtered.strings' % (short_vector_dataset_name, composer.name))
        features_path = os.path.join(output_dir,
                                     'AN_NN_%s_%s.features.filtered.strings' % (
                                         short_vector_dataset_name, composer.name))

        rows2idx = {i: DocumentFeature.from_string(x) for (x, i) in rows.items()}
        write_vectors_to_disk(mat.tocoo(), rows2idx, cols, events_path,
                              features_path=features_path, entries_path=entries_path,
                              entry_filter=lambda x: x.type in {'AN', 'NN', '1-GRAM'})


classification_data_path = ['/mnt/lustre/scratch/inf/mmb28/thesisgenerator/sample-data/reuters21578/r8-tagged-grouped']

classification_data_path_mr = ['/mnt/lustre/scratch/inf/mmb28/thesisgenerator/sample-data/movie-reviews-tagged']

technion_data_paths = glob('/mnt/lustre/scratch/inf/mmb28/thesisgenerator/sample-data/techtc100-clean/*')

all_classification_corpora = classification_data_path + classification_data_path_mr + technion_data_paths