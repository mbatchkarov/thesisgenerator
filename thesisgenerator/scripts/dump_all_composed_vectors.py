from glob import glob
import os
import sys
from discoutils.misc import is_gzipped

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from thesisgenerator.composers.vectorstore import *
from thesisgenerator.scripts.extract_NPs_from_labelled_data import get_all_NPs
from discoutils.io_utils import write_vectors_to_disk


def compose_and_write_vectors(unigram_vectors_path, short_vector_dataset_name,
                              composer_classes, pretrained_Baroni_composer_file=None,
                              output_dir='.', gzipped=True):
    """
    Extracts all composable features from a labelled classification corpus and dumps a composed vector for each of them
    to disk. The output file will also contain all unigram vectors that were passed in, and only unigrams!
    :param unigram_vectors_path: a file in Byblo events format that contain vectors for all unigrams OR
    a Vectors object. This will be used in the composition process
    :type unigram_vectors_path: str or Vectors
    :param classification_corpora: Corpora to extract features from. Dict {corpus_path: conf_file}
    :param pretrained_Baroni_composer_file: path to pre-trained Baroni AN/NN composer file
    :param output_dir:
    :param composer_classes: what composers to use
    :type composer_classes: list
    """

    vocabulary = get_all_NPs()
    if isinstance(unigram_vectors_path, Vectors):
        vectors = unigram_vectors_path
    else:
        # ensure there's only unigrams in the set of unigram vectors
        # composers do not need any ngram vectors contain in this file, they may well be
        # observed ones
        vectors = Vectors.from_tsv(unigram_vectors_path,
                                   row_filter=lambda x, y: y.tokens[0].pos in {'N', 'J'} and y.type == '1-GRAM',
                                   gzipped=is_gzipped(unigram_vectors_path))

    # doing this loop in parallel isn't worth it as pickling or shelving `vectors` is so slow
    # it negates any gains from using multiple cores
    for composer_class in composer_classes:
        if composer_class == BaroniComposer:
            assert pretrained_Baroni_composer_file is not None
            composer = BaroniComposer(vectors, pretrained_Baroni_composer_file)
        else:
            composer = composer_class(vectors)

        # compose_all returns all unigrams and composed phrases
        mat, cols, rows = composer.compose_all(vocabulary)

        events_path = os.path.join(output_dir,
                                   'AN_NN_%s_%s.events.filtered.strings' % (short_vector_dataset_name, composer.name))
        rows2idx = {i: DocumentFeature.from_string(x) for (x, i) in rows.items()}
        write_vectors_to_disk(mat.tocoo(), rows2idx, cols, events_path,
                              entry_filter=lambda x: x.type in {'AN', 'NN', '1-GRAM'},
                              gzipped=gzipped)