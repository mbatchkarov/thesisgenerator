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


def _loop_body(composer_class, output_dir, pipeline, pretrained_Baroni_composer_files, short_vector_dataset_name,
               unigram_source, x_ev, x_tr, y_ev, y_tr):
    # whatever files we need to (entries index, as composition creates new entries)
    if composer_class == BaroniComposer:
        composers = [BaroniComposer(unigram_source, path) for path in pretrained_Baroni_composer_files]
    else:
        composers = [composer_class(unigram_source)]
    vector_source = CompositeVectorSource(composers, sim_threshold=0, include_self=False)
    fit_args = {
        'stripper__vector_source': vector_source,
        'vect__vector_source': vector_source,
        'fs__vector_source': vector_source,
    }
    _ = pipeline.fit_transform(x_tr + x_ev, y=hstack([y_tr, y_ev]), **fit_args)

    output_files = ('AN_NN_%s_%s.events.filtered.strings' % ( short_vector_dataset_name, composers[0].name),
                    'AN_NN_%s_%s.entries.filtered.strings' % ( short_vector_dataset_name, composers[0].name),
                    'AN_NN_%s_%s.features.filtered.strings' % ( short_vector_dataset_name, composers[0].name))
    output_files = [os.path.join(output_dir, x) for x in output_files]

    pipeline.steps[2][1].vector_source.write_vectors_to_disk({'AN', 'NN'}, *output_files)


def compose_and_write_vectors(unigram_vector_paths, short_vector_dataset_name, classification_data_paths,
                              pretrained_Baroni_composer_files,
                              output_dir='.', composer_classes='bar'):
    """
    Extracts all composable features from a labelled classification corpus and dumps a vector for each of them
    to disk
    :param unigram_vector_paths: a list of files in Byblo events format that contain vectors for all unigrams. This
    will be used in the composition process
    :param classification_data_paths:
    :param pretrained_Baroni_composer_files: path to pre-trained Baroni AN composer file
    :type pretrained_Baroni_composer_files: list
    :param output_dir:
    :param composer_classes:
    :return:
    :rtype: list of strings
    """
    train, test = classification_data_paths
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
    tokenised_data = tokenize_data(raw_data, tokenizer, data_ids)

    pipeline = Pipeline([
        ('vect', ThesaurusVectorizer(ngram_range=(0, 0), min_df=1, use_tfidf=False)),
        ('fs', VectorBackedSelectKBest(ensure_vectors_exist=True)),
        ('stripper', MetadataStripper(nn_algorithm='brute', build_tree=False))
    ])
    x_tr, y_tr, x_ev, y_ev = tokenised_data
    unigram_source = UnigramVectorSource(unigram_vector_paths, reduce_dimensionality=False)
    Parallel(n_jobs=7)(delayed(_loop_body)(composer_class,
                                           output_dir,
                                           pipeline,
                                           pretrained_Baroni_composer_files,
                                           short_vector_dataset_name,
                                           unigram_source,
                                           x_ev, x_tr,
                                           y_ev, y_tr) for composer_class in composer_classes)

    #for composer_class in composer_classes:
    #    _loop_body(composer_class, output_dir, p, pretrained_Baroni_composer_files,
    #               short_vector_dataset_name, unigram_source, x_ev, x_tr, y_ev, y_tr)


giga_paths = [
    '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp6-12a/exp6.events.filtered.strings',
    '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp6-12b/exp6.events.filtered.strings',
    '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp6-12c/exp6.events.filtered.strings',
    '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp6-12d/exp6.events.filtered.strings'
]

wiki_paths = [
    '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/wikipedia_t100/wiki_t100f100_adjs_deps/wikipedia_adjsdeps_t100.pbfiltered.events.strings',
    '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/wikipedia_t100/wiki_t100f100_advs_deps/wikipedia_advsdeps_t100.pbfiltered.events.strings',
    '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/wikipedia_t100/wiki_t100f100_verbs_deps/wikipedia_verbsdeps_t100.pbfiltered.events.strings',
    '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/wikipedia_t100/wiki_t100f100_nouns_deps/wikipedia_nounsdeps_t100.pbfiltered.events.strings',
]

toy_paths = [
    '/Volumes/LocalDataHD/mmb28/NetBeansProjects/Byblo-2.2.0/sample-data/7head.txt.events.filtered.strings'
]

n_jobs = 4
classification_data_path = (
    '/mnt/lustre/scratch/inf/mmb28/thesisgenerator/sample-data/reuters21578/r8train-tagged-grouped',
    '/mnt/lustre/scratch/inf/mmb28/thesisgenerator/sample-data/reuters21578/r8test-tagged-grouped')

classification_data_path_mr = (
    '/mnt/lustre/scratch/inf/mmb28/thesisgenerator/sample-data/movie-reviews-train-tagged',
    '/mnt/lustre/scratch/inf/mmb28/thesisgenerator/sample-data/movie-reviews-test-tagged')

if __name__ == '__main__':
    """
    Call with any command-line parameters to enable debug mode
    """


    #composers = [AdditiveComposer, MultiplicativeComposer]
    #vector_paths = [giga_paths, wiki_paths]
    #vector_paths = [toy_paths]
    vector_paths = [giga_paths]
    trained_baroni_model = '/Volumes/LocalDataHD/mmb28/NetBeansProjects/FeatureExtractionToolkit/phrases/julie.ANs.clean.AN-model.model.pkl'

    debug = len(sys.argv) > 1
    if debug:
        #giga_paths.pop(0)
        #giga_paths.pop(0)
        #wiki_paths.pop(-1)
        #wiki_paths.pop(-1)
        n_jobs = 1
        #data_path = ['%s-small' % corpus_path for corpus_path in data_path]

    output_files = Parallel(n_jobs=n_jobs)(
        delayed(compose_and_write_vectors)(vectors_path, classification_data_path, trained_baroni_model,
                                           log_to_console=debug)
        for vectors_path in vector_paths)
    for vectors, entries, features in output_files:
        print vectors, entries, features

