from glob import glob
import os
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from sklearn.pipeline import Pipeline
from thesisgenerator.composers.feature_selectors import VectorBackedSelectKBest
from thesisgenerator.composers.vectorstore import *
from thesisgenerator.plugins.bov import ThesaurusVectorizer
from thesisgenerator.utils.data_utils import get_tokenized_data, get_tokenizer_settings_from_conf_file
from discoutils.io_utils import write_vectors_to_disk


def compose_and_write_vectors(unigram_vectors_path, short_vector_dataset_name, classification_corpora,
                              composer_classes, pretrained_Baroni_composer_file=None,
                              output_dir='.'):
    """
    Extracts all composable features from a labelled classification corpus and dumps a composed vector for each of them
    to disk. The output file will also contain all unigram vectors that were passed in.
    :param unigram_vectors_path: a file in Byblo events format that contain vectors for all unigrams. This
    will be used in the composition process
    :param classification_corpora: Corpora to extract features from. Dict {corpus_path: conf_file}
    :type classification_corpora: dict
    :param pretrained_Baroni_composer_file: path to pre-trained Baroni AN/NN composer file
    :param output_dir:
    :param composer_classes: what composers to use
    :type composer_classes: list
    """
    all_text = []
    for corpus_path, conf_file in classification_corpora.items():
        x_tr, _, _, _ = get_tokenized_data(corpus_path + '.gz',
                                           get_tokenizer_settings_from_conf_file(conf_file))
        assert len(x_tr) > 0
        all_text.extend(x_tr)
        logging.info('Documents so far: %d', len(all_text))

    pipeline = Pipeline([
        # do not extract unigrams explicitly, all unigrams that we have vectors for will be written anyway (see below)
        ('vect', ThesaurusVectorizer(ngram_range=(0, 0), min_df=1, use_tfidf=False,
                                     extract_VO_features=False, extract_SVO_features=False,
                                     unigram_feature_pos_tags=['N', 'J'])),
        ('fs', VectorBackedSelectKBest(must_be_in_thesaurus=True)),
    ])

    vectors = Vectors.from_tsv(unigram_vectors_path,
                               # ensure there's only unigrams in the set of unigram vectors
                               # composers do not need any ngram vectors contain in this file, they may well be
                               # observed ones
                               row_filter=lambda x, y: y.tokens[0].pos in {'N', 'J'} and y.type == '1-GRAM')

    # doing this loop in parallel isn't worth it as pickling or shelving `vectors` is so slow
    # it negates any gains from using multiple cores
    for composer_class in composer_classes:
        if composer_class == BaroniComposer:
            assert pretrained_Baroni_composer_file is not None
            composer = BaroniComposer(vectors, pretrained_Baroni_composer_file)
        else:
            composer = composer_class(vectors)

        fit_args = {
            'vect__vector_source': composer,
            'fs__vector_source': composer,
        }
        # y doesn't really matter here, but is required by fit_transform
        _, vocabulary = pipeline.fit_transform(all_text, y=np.arange(len(all_text)), **fit_args)
        logging.info('Found a total of %d document features', len(vocabulary))
        # compose_all returns all unigrams and composed phrases
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