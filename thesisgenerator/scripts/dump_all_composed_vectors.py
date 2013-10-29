import logging
from itertools import product

from joblib import Parallel, delayed
from numpy import hstack
from sklearn.pipeline import Pipeline

from thesisgenerator.composers.feature_selectors import VectorBackedSelectKBest, MetadataStripper
from thesisgenerator.composers.vectorstore import UnigramVectorSource, UnigramDummyComposer, AdditiveComposer, CompositeVectorSource, MultiplicativeComposer
from thesisgenerator.plugins.bov import ThesaurusVectorizer
from thesisgenerator.utils.data_utils import load_tokenizer, tokenize_data, load_text_data_into_memory


def do_work(unigram_paths, composer_class):
    # todo should take a directory containing a thesaurus so that we can read vectors from events file and modify
    # whatever files we need to (entries index, as composition creates new entries)
    unigram_source = UnigramVectorSource(unigram_paths, reduce_dimensionality=False)
    composers = [
        UnigramDummyComposer(unigram_source),
        composer_class(unigram_source)
    ]
    vector_source = CompositeVectorSource(composers, 0, False)

    raw_data, data_ids = load_text_data_into_memory(
        training_path='sample-data/reuters21578/r8train-tagged-grouped',
        test_path='sample-data/reuters21578/r8test-tagged-grouped',
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
        ('vect', ThesaurusVectorizer(ngram_range=(1, 2), min_df=1, use_tfidf=False)),
        ('fs', VectorBackedSelectKBest(ensure_vectors_exist=True)),
        ('stripper', MetadataStripper())
    ])
    x_tr, y_tr, x_ev, y_ev = tokenised_data

    fit_args = {
        'stripper__vector_source': vector_source,
        'vect__vector_source': vector_source,
        'fs__vector_source': vector_source,
    }
    _ = p.fit_transform(x_tr + x_ev, y=hstack([y_tr, y_ev]), **fit_args)

    dataset = 'wiki' if 'wiki' in unigram_paths[0] else 'gigaw'
    composer_method = composer_class.__name__[:4]
    p.steps[2][1].vector_source.dump_vectors('bigram_%s_%s.vectors.tsv' % (dataset, composer_method),
                                             'bigram_%s_%s.entries.txt' % (dataset, composer_method),
                                             'bigram_%s_%s.features.txt' % (dataset, composer_method))


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s\t%(module)s.%(funcName)s ""(line"
                           " %(lineno)d)\t%(levelname)s : %(""message)s"
)

giga_paths = [
    '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp6-12a/exp6.events.strings',
    '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp6-12b/exp6.events.strings',
    '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp6-12c/exp6.events.strings',
    '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp6-12d/exp6.events.strings'
]

wiki_paths = [
    '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/wikipedia_t100/wiki_t100f100_adjs_deps/wikipedia_adjsdeps_t100.pbfiltered.events.strings',
    '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/wikipedia_t100/wiki_t100f100_advs_deps/wikipedia_advsdeps_t100.pbfiltered.events.strings',
    '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/wikipedia_t100/wiki_t100f100_verbs_deps/wikipedia_verbsdeps_t100.pbfiltered.events.strings'
    '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/wikipedia_t100/wiki_t100f100_nouns_deps/wikipedia_nounsdeps_t100.pbfiltered.events.strings',
]

composers = [AdditiveComposer, MultiplicativeComposer]
paths = [giga_paths, wiki_paths]

Parallel(n_jobs=4)(delayed(do_work)(path, composer) for composer, path in product(composers, paths))
