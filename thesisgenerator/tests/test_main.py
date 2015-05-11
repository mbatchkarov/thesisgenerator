# coding=utf-8
import glob
import os
import numpy as np
from numpy.ma import std
import numpy.testing as t
import pytest
import scipy.sparse as sp
from discoutils.thesaurus_loader import Thesaurus
from thesisgenerator.composers.vectorstore import DummyThesaurus

from thesisgenerator import __main__
from thesisgenerator.tests.test_feature_selectors import strip
from thesisgenerator.utils.data_utils import get_tokenized_data, jsonify_single_labelled_corpus


tokenizer_opts = {
    'normalise_entities': False,
    'use_pos': True,
    'coarse_pos': True,
    'lemmatize': True,
    'lowercase': True,
    'remove_stopwords': False,
    'remove_short_words': False
}

training_matrix = np.array([
    [1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1],
])

pruned_training_matrix = np.array([
    [1, 1, 0],
    [0, 0, 1],
])
pruned_vocab = {'a/N': 0, 'b/N': 1, 'd/N': 2}
full_vocab = {'a/N': 0, 'b/N': 1, 'c/N': 2, 'd/N': 3, 'e/N': 4, 'f/N': 5}


@pytest.fixture
def feature_extraction_conf():
    return {
        'vectorizer': 'thesisgenerator.plugins.bov.ThesaurusVectorizer',
        'analyzer': 'ngram',
        'use_tfidf': False,
        'min_df': 1,
        'lowercase': False,
        'record_stats': True,
        'k': 10,  # use all thesaurus entries
        'train_token_handler': 'thesisgenerator.plugins.bov_feature_handlers.BaseFeatureHandler',
        'decode_token_handler': 'thesisgenerator.plugins.bov_feature_handlers.BaseFeatureHandler',
        'train_time_opts': dict(extract_unigram_features=['J', 'N', 'V'],
                                extract_phrase_features=[]),
        'decode_time_opts': dict(extract_unigram_features=['J', 'N', 'V'],
                                 extract_phrase_features=[])
    }


@pytest.fixture(scope='module')
def feature_selection_conf():
    return {
        'run': True,
        'method': 'thesisgenerator.composers.feature_selectors.VectorBackedSelectKBest',
        'scoring_function': 'sklearn.feature_selection.chi2',
        'must_be_in_thesaurus': False,
        'k': 'all',
        'thesaurus': None
    }


def _vectorize_data(data_paths, feature_selection_conf=feature_selection_conf(),
                    feature_extraction_conf=feature_extraction_conf(), vector_source=None):
    # at this point _load_data should have been called and as a result the fields
    # x_tr, y_tr, x_test and y_test must have been initialised
    # also, tokenizer and thesaurus must have been initialised
    if isinstance(vector_source, str):
        vector_source = Thesaurus.from_tsv(vector_source)

    feature_selection_conf['thesaurus'] = vector_source
    pipeline, fit_params = __main__._build_pipeline(
        12345,  # id for naming debug files
        vector_source,
        # None, # classifier
        feature_extraction_conf,
        feature_selection_conf,
        # None, # classifier options
        '.',  # temp files dir
        True,  # debug mode
        'test_main'  # name of experiments
    )

    x_tr, y_tr, x_test, y_test = get_tokenized_data(data_paths[0], tokenizer_opts, test_data=data_paths[1])

    x1 = pipeline.fit_transform(x_tr, y_tr, **fit_params)
    if 'fs' in pipeline.named_steps:
        pipeline.named_steps['vect'].vocabulary_ = pipeline.named_steps['fs'].vocabulary_

    voc = pipeline.named_steps['fs'].vocabulary_
    x2 = pipeline.transform(x_test)

    return x1, x2, voc


def tearDown():
    for pattern in ['PostVectDump_test_main*', 'stats-test_main-cv12345*']:
        files = glob.glob(pattern)
        for f in files:
            if os.path.exists(f):
                os.remove(f)


@pytest.fixture(params=['xml', 'json'], scope='module')
def data(request):
    """
    Returns path to a labelled dataset on disk
    """
    kind = request.param
    prefix = 'thesisgenerator/resources/test-baseline'
    tr_path = '%s-tr' % prefix
    ev_path = '%s-ev' % prefix

    request.addfinalizer(tearDown)

    if kind == 'xml':
        # return the raw corpus in XML
        return tr_path, ev_path
    if kind == 'json':
        # convert corpus to gzipped JSON and try again
        jsonify_single_labelled_corpus(tr_path, tokenizer_conf=tokenizer_opts)
        jsonify_single_labelled_corpus(ev_path, tokenizer_conf=tokenizer_opts)
        return tr_path + '.gz', ev_path + '.gz'


def test_baseline_use_all_features_signifier_only(data):
    tsv_file = 'thesisgenerator/resources/exp0-0b.strings'

    x1, x2, voc = _vectorize_data(data, vector_source=tsv_file)
    assert full_vocab == strip(voc)

    assert isinstance(x1, sp.spmatrix)
    t.assert_array_equal(
        x1.toarray(),
        training_matrix
    )

    t.assert_array_equal(
        x2.toarray(),
        np.array(
            [
                [4, 1, 2, 0, 0, 0],
            ]
        )
    )


def test_baseline_ignore_nonthesaurus_features_signifier_only(data, feature_selection_conf):
    feature_selection_conf['must_be_in_thesaurus'] = True
    tsv_file = 'thesisgenerator/resources/exp0-0b.strings'

    x1, x2, voc = _vectorize_data(data,
                                  feature_selection_conf=feature_selection_conf,
                                  vector_source=tsv_file)

    assert pruned_vocab == strip(voc)

    # assertIsInstance(x1, sp.spmatrix)
    t.assert_array_equal(
        x1.toarray(),
        pruned_training_matrix
    )

    t.assert_array_equal(
        x2.toarray(),
        np.array(
            [
                [4, 1, 0]
            ]
        )
    )


def test_baseline_use_all_features_with__signifier_signified(data, feature_extraction_conf,
                                                             feature_selection_conf):
    feature_selection_conf['must_be_in_thesaurus'] = False
    feature_extraction_conf['decode_token_handler'] = \
        'thesisgenerator.plugins.bov_feature_handlers.SignifierSignifiedFeatureHandler'
    feature_extraction_conf['k'] = 1  # equivalent to max
    tsv_file = 'thesisgenerator/resources/exp0-0b.strings'

    x1, x2, voc = _vectorize_data(data,
                                  feature_extraction_conf=feature_extraction_conf,
                                  feature_selection_conf=feature_selection_conf,
                                  vector_source=tsv_file)

    assert full_vocab == strip(voc)

    assert isinstance(x1, sp.spmatrix)
    t.assert_array_equal(
        x1.toarray(),
        training_matrix
    )

    t.assert_array_almost_equal(
        x2.toarray(),
        np.array(
            [
                [4, 1, 2, 2.1, 0, 0]
            ]
        )
    )


def test_baseline_ignore_nonthesaurus_features_with_signifier_signified(data, feature_extraction_conf,
                                                                        feature_selection_conf):
    feature_selection_conf['must_be_in_thesaurus'] = True
    feature_extraction_conf['decode_token_handler'] = \
        'thesisgenerator.plugins.bov_feature_handlers.SignifierSignifiedFeatureHandler'
    feature_extraction_conf['k'] = 1
    tsv_file = 'thesisgenerator/resources/exp0-0b.strings'

    x1, x2, voc = _vectorize_data(data,
                                  feature_extraction_conf=feature_extraction_conf,
                                  feature_selection_conf=feature_selection_conf,
                                  vector_source=tsv_file)
    assert pruned_vocab == strip(voc)
    assert isinstance(x1, sp.spmatrix)
    t.assert_array_equal(
        x1.toarray(),
        pruned_training_matrix
    )

    t.assert_array_almost_equal(
        x2.toarray(),
        np.array(
            [
                [4, 1, 2.1]
            ]
        )
    )


def test_baseline_use_all_features_with_signified(data, feature_extraction_conf, feature_selection_conf):
    feature_selection_conf['must_be_in_thesaurus'] = False
    feature_extraction_conf['decode_token_handler'] = \
        'thesisgenerator.plugins.bov_feature_handlers.SignifiedOnlyFeatureHandler'
    feature_extraction_conf['k'] = 1  # equivalent to max
    tsv_file = 'thesisgenerator/resources/exp0-0b.strings'

    x1, x2, voc = _vectorize_data(data,
                                  feature_extraction_conf=feature_extraction_conf,
                                  feature_selection_conf=feature_selection_conf,
                                  vector_source=tsv_file)

    assert full_vocab == strip(voc)

    assert isinstance(x1, sp.spmatrix)
    t.assert_array_equal(
        x1.toarray(),
        training_matrix
    )

    t.assert_array_almost_equal(
        x2.toarray(),
        np.array(
            [
                [0, 0, 0, 4.4, 0, 0],
            ]
        )
    )


def test_baseline_ignore_nonthesaurus_features_with_signified(data, feature_extraction_conf, feature_selection_conf):
    feature_selection_conf['must_be_in_thesaurus'] = True
    feature_extraction_conf['decode_token_handler'] = \
        'thesisgenerator.plugins.bov_feature_handlers.SignifiedOnlyFeatureHandler'
    feature_extraction_conf['k'] = 1  # equivalent to max
    tsv_file = 'thesisgenerator/resources/exp0-0b.strings'

    x1, x2, voc = _vectorize_data(data,
                                  feature_extraction_conf=feature_extraction_conf,
                                  feature_selection_conf=feature_selection_conf,
                                  vector_source=tsv_file)

    assert pruned_vocab == strip(voc)

    assert isinstance(x1, sp.spmatrix)
    t.assert_array_equal(
        x1.toarray(),
        pruned_training_matrix
    )

    t.assert_array_almost_equal(
        x2.toarray(),
        np.array(
            [
                [0, 0, 4.4]
            ]
        )
    )


def test_baseline_use_all_features_with_signified_random(data, feature_extraction_conf, feature_selection_conf):
    feature_selection_conf['must_be_in_thesaurus'] = False
    feature_extraction_conf['decode_token_handler'] = \
        'thesisgenerator.plugins.bov_feature_handlers.SignifierRandomBaselineFeatureHandler'
    feature_extraction_conf['k'] = 1
    feature_extraction_conf['neighbour_source'] = \
        'thesisgenerator.tests.test_main._get_constant_thesaurus'

    source = DummyThesaurus()
    x1, x2, voc = _vectorize_data(data,
                                  feature_extraction_conf=feature_extraction_conf,
                                  feature_selection_conf=feature_selection_conf,
                                  vector_source=source)

    assert full_vocab == strip(voc)

    assert isinstance(x1, sp.spmatrix)
    t.assert_array_equal(
        x1.toarray(),
        training_matrix
    )

    t.assert_array_almost_equal(
        x2.toarray(),
        np.array(
            [
                [0, 11.0, 0, 0, 0, 0],
            ]
        )
    )
    # the thesaurus will always say the neighbour for something is
    # b/N with a similarity of 1, and we look up 11 tokens overall in
    # the test document
    source.vocab = voc
    x1, x2, voc = _vectorize_data(data, vector_source=source)
    assert x2.sum(), 11.0
    assert std(x2.todense()) > 0
    # seven tokens will be looked up, with random in-vocabulary neighbours
    # returned each time. Std>0 shows that it's not the same thing
    # returned each time
    # print x2