import logging
import os
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from joblib import Parallel, delayed
from numpy import hstack
from sklearn.pipeline import Pipeline
from thesisgenerator.composers.feature_selectors import VectorBackedSelectKBest, MetadataStripper
from thesisgenerator.composers.vectorstore import UnigramVectorSource, UnigramDummyComposer, \
    CompositeVectorSource, BaroniComposer, OxfordSvoComposer
from thesisgenerator.plugins.bov import ThesaurusVectorizer
from thesisgenerator.utils.data_utils import load_tokenizer, tokenize_data, load_text_data_into_memory


def write_vectors(unigram_paths, data_paths, output_dir='.', log_to_console=False):
    if 'wiki' in unigram_paths[0]:
        dataset = 'wiki'
    elif '7head' in unigram_paths[0]:
        dataset = '7head'
    else:
        dataset = 'gigaw'
        #composer_method = composer_class.__name__[:4]
    composer_method = 'bar_svo'

    output_files = ('bigram_%s_%s.vectors.tsv' % (dataset, composer_method),
                    'bigram_%s_%s.entries.txt' % (dataset, composer_method),
                    'bigram_%s_%s.features.txt' % (dataset, composer_method))
    output_files = [os.path.join(output_dir, x) for x in output_files]

    if all(os.path.exists(f) for f in output_files):
        return output_files

    params = dict(
        filename='bigram_%s_%s.log' % (dataset, composer_method),
        level=logging.DEBUG,
        format="%(asctime)s\t%(module)s.%(funcName)s ""(line %(lineno)d)\t%(levelname)s : %(""message)s"
    )
    if log_to_console:
        del params['filename']
    logging.basicConfig(**params)


    # todo should take a directory containing a thesaurus so that we can read vectors from events file and modify
    # whatever files we need to (entries index, as composition creates new entries)
    unigram_source = UnigramVectorSource(unigram_paths, reduce_dimensionality=False)
    composers = [
        UnigramDummyComposer(unigram_source),
        BaroniComposer(unigram_source),
        OxfordSvoComposer(unigram_source),
        #composer_class(unigram_source)
    ]
    vector_source = CompositeVectorSource(composers, 0, False)

    train, test = data_paths
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

    p = Pipeline([
        ('vect', ThesaurusVectorizer(ngram_range=(0, 0), min_df=1, use_tfidf=False)),
        ('fs', VectorBackedSelectKBest(ensure_vectors_exist=True)),
        ('stripper', MetadataStripper(nn_algorithm='brute', build_tree=False))
    ])
    x_tr, y_tr, x_ev, y_ev = tokenised_data

    fit_args = {
        'stripper__vector_source': vector_source,
        'vect__vector_source': vector_source,
        'fs__vector_source': vector_source,
    }
    _ = p.fit_transform(x_tr + x_ev, y=hstack([y_tr, y_ev]), **fit_args)

    p.steps[2][1].vector_source.write_vectors_to_disk(*output_files)
    return output_files


giga_paths = [
    '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp6-12a/exp6.events.strings',
    '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp6-12b/exp6.events.strings',
    '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp6-12c/exp6.events.strings',
    '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp6-12d/exp6.events.strings'
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
data_path = ('/mnt/lustre/scratch/inf/mmb28/thesisgenerator/sample-data/reuters21578/r8train-tagged-grouped',
             '/mnt/lustre/scratch/inf/mmb28/thesisgenerator/sample-data/reuters21578/r8test-tagged-grouped')

if __name__ == '__main__':
    """
    Call with any command-line parameters to enable debug mode
    """


    #composers = [AdditiveComposer, MultiplicativeComposer]
    #vector_paths = [giga_paths, wiki_paths]
    vector_paths = [toy_paths]

    debug = len(sys.argv) > 1
    if debug:
        #giga_paths.pop(0)
        #giga_paths.pop(0)
        #wiki_paths.pop(-1)
        #wiki_paths.pop(-1)
        n_jobs = 1
        data_path = ['%s-small' % corpus_path for corpus_path in data_path]

    output_files = Parallel(n_jobs=n_jobs)(delayed(write_vectors)(vectors_path, data_path, log_to_console=debug)
                                           for vectors_path in vector_paths)
    for vectors, entries, features in output_files:
        print vectors, entries, features

