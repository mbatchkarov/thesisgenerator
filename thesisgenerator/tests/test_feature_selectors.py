import os

from sklearn.pipeline import Pipeline
import numpy as np
import numpy.testing as t
from pandas.io.parsers import read_csv

from thesisgenerator.composers.feature_selectors import VectorBackedSelectKBest
from thesisgenerator.composers.vectorstore import CompositeVectorSource, UnigramVectorSource, \
    UnigramDummyComposer, PrecomputedSimilaritiesVectorSource
from thesisgenerator.plugins.bov import ThesaurusVectorizer
from thesisgenerator.plugins.dumpers import FeatureVectorsCsvDumper
from discoutils.tokens import DocumentFeature, Token
from thesisgenerator.utils.data_utils import load_text_data_into_memory, load_tokenizer, tokenize_data


training_matrix_signifier_bigrams = np.array(
    [[1., 1., 0., 0., 1., 0., 1., 0., 1., 0.],
     [1., 1., 0., 0., 1., 0., 1., 0., 1., 0.],
     [0., 0., 1., 1., 0., 1., 0., 1., 0., 1.]])


def strip(mydict):
    """{ DocumentFeature('1-GRAM', ('X', 'Y',)) : int} -> {'X Y' : int}"""
    return {feature.tokens_as_str(): count for feature, count in mydict.iteritems()}


def _do_feature_selection(must_be_in_thesaurus, k, handler='Base', vector_source='default', max_feature_len=1,
                          delete_kid=False):
    """
    Loads a data set, vectorizes it by extracting n-grams (default n=1) using a feature handler (default
    BaseFeatureHandler) and then performs feature selection based on either a vector source or on chi2 scores.
    Returns the encode/decode matrices and the stripped vocabulary of the Vectorizer after feature selection.

    The vector source by default has a unigrams source that covers all unigrams in the training set
    (feature vectors are made up), and does not know about n-grams. Optionally, another vector
    source can be passed in.
    """
    handler_pattern = 'thesisgenerator.plugins.bov_feature_handlers.{}FeatureHandler'
    raw_data, data_ids = load_text_data_into_memory(
        training_path='thesisgenerator/resources/test-tr',
        test_path='thesisgenerator/resources/test-ev',
    )

    tokenizer = load_tokenizer()
    x_train, y_train, x_test, y_test = tokenize_data(raw_data, tokenizer, data_ids)

    if vector_source == 'default':
        unigrams_vectors = UnigramVectorSource(
            ['thesisgenerator/resources/thesauri/exp0-0a.txt.events-unfiltered.strings'])
        vector_source = CompositeVectorSource([UnigramDummyComposer(unigrams_vectors)],
                                              0.0, False)

    if delete_kid:
        # the set of vectors we load from disk covers all unigrams in the training set, which makes it boring
        # let's remove one entry
        del unigrams_vectors.entry_index[DocumentFeature.from_string('kid/N')]
        unigrams_vectors.feature_matrix = unigrams_vectors.feature_matrix[:, :-1]

    pipeline_list = [
        ('vect',
         ThesaurusVectorizer(min_df=1, use_tfidf=False,
                             ngram_range=(1, max_feature_len),
                             decode_token_handler=handler_pattern.format(handler))),
        ('fs', VectorBackedSelectKBest(must_be_in_thesaurus=must_be_in_thesaurus, k=k)),
        ('dumper', FeatureVectorsCsvDumper('fs-test'))
    ]
    p = Pipeline(pipeline_list)
    fit_params = {'vect__vector_source': vector_source,
                  'fs__vector_source': vector_source}

    tr_matrix, tr_voc = p.fit_transform(x_train, y_train, **fit_params)
    if 'fs' in p.named_steps:
        p.named_steps['vect'].vocabulary_ = p.named_steps['fs'].vocabulary_
    ev_matrix, ev_voc = p.transform(x_test)
    return tr_matrix.A, strip(tr_voc), ev_matrix.A, strip(ev_voc)


def test_unigrams_without_feature_selection():
    """
    Test without any feature selection and unigram features only, matrices and vocabulary are as in
    test_main.test_baseline_use_all_features_signifier_only.
    """

    # training corpus is "cats like dogs" (x2), "kids play games"
    # eval corpus is "birds like fruit" (x2), "dogs eat birds"
    # thesaurus contains cat, dog, game, kid, fruit, like, play
    tr_matrix, tr_voc, ev_matrix, ev_voc = _do_feature_selection(False, 'all', vector_source=None)
    # should work without a vector source, because we use signified encoding only and no vector-based FS
    voc = {
        'cat/N': 0,
        'dog/N': 1,
        'game/N': 2,
        'kid/N': 3,
        'like/V': 4,
        'play/V': 5
    }
    assert tr_voc == ev_voc
    assert tr_voc == voc

    t.assert_array_equal(tr_matrix, np.array(
        [[1., 1., 0., 0., 1., 0.],
         [1., 1., 0., 0., 1., 0.],
         [0., 0., 1., 1., 0., 1.]]))
    t.assert_array_equal(ev_matrix, np.array(
        [[0., 0., 0., 0., 1., 0.],
         [0., 0., 0., 0., 1., 0.],
         [0., 1., 0., 0., 0., 0.]]))
    _check_debug_file(ev_matrix, tr_matrix, voc)


def test_with_thesaurus_feature_selection_only():
    """
    Tests if features in the training data not contained in the vector source are removed. A feature is
    removed from the default vector source.
    """
    tr_matrix, tr_voc, ev_matrix, ev_voc = _do_feature_selection(True, 'all', delete_kid=True)

    voc = {
        'cat/N': 0,
        'dog/N': 1,
        'game/N': 2,
        #'kid/N': 3, # removed because vector is missing, this happens in _do_feature_selection
        'like/V': 3,
        'play/V': 4
    }
    assert tr_voc == voc
    assert tr_voc == ev_voc

    t.assert_array_equal(tr_matrix,
                         np.array(
                             [[1., 1., 0., 1., 0.],
                              [1., 1., 0., 1., 0.],
                              [0., 0., 1., 0., 1.]]))
    t.assert_array_equal(ev_matrix,
                         np.array(
                             [[0., 0., 0., 1., 0.],
                              [0., 0., 0., 1., 0.],
                              [0., 1., 0., 0., 0.]]))
    _check_debug_file(ev_matrix, tr_matrix, voc)


def test_unigrams_with_chi2_feature_selection_only():
    """
    Test the textbook case of feature selection, where some number of features are removed because they are
    not informative. Unigram features only, no vector source needed.
    """
    tr_matrix, tr_voc, ev_matrix, ev_voc = _do_feature_selection(False, 3, vector_source=None)

    # feature scores at train time are [ 1.  1.  2.  2.  1.  2.]. These are provided by sklearn and I have not
    # verified them. Higher seems to be better (the textbook implementation of chi2 says lower is better)
    voc = {
        #'cat/N': 0, # removed because their chi2 score is low
        #'dog/N': 1,
        'game/N': 0,
        'kid/N': 1,
        #'like/V': 4,
        'play/V': 2
    }
    assert tr_voc == voc
    assert tr_voc == ev_voc

    t.assert_array_equal(tr_matrix,
                         np.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [1., 1., 1.]]))

    t.assert_array_equal(ev_matrix,
                         np.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]]))
    _check_debug_file(ev_matrix, tr_matrix, voc)


def test_with_chi2_and_thesaurus_feature_selection():
    """
    Test a combination of feature selection through vector source and low chi2 score. Unigram features only.
    """
    tr_matrix, tr_voc, ev_matrix, ev_voc = _do_feature_selection(True, 3, delete_kid=True)

    assert tr_voc == ev_voc
    voc = {
        #'cat/N': 0, # removed because of low chi2 score
        #'dog/N': 1,  # removed because of low chi2 score
        'game/N': 0,
        #'kid/N': 3, # removed because vector is missing
        #'like/V': 4,  # removed because of low chi2 score
        'play/V': 1
    }
    # feature scores at train time are [ 1.  1.  2.  2.  1.  2.]
    assert tr_voc == voc

    t.assert_array_equal(tr_matrix, np.array([[0., 0.],
                                              [0., 0.],
                                              [1., 1.]]))

    t.assert_array_equal(ev_matrix, np.array([[0., 0.],
                                              [0., 0.],
                                              [0., 0.]]))

    _check_debug_file(ev_matrix, tr_matrix, voc)


def test_simple_bigram_features_without_fs():
    """
    A standard textbook setup with a limited number of useful bigram features, no feature selection of the basis
    of vectors. No vector source needed.
    """
    tr_matrix, tr_voc, ev_matrix, ev_voc = _do_feature_selection(False, 'all', vector_source=None,
                                                                 max_feature_len=2)
    assert ev_matrix.shape == (3, 10)

    # vocabulary sorted by feature length and then alphabetically-- default behaviour of python's sorted()
    assert tr_voc == {'cat/N': 0,
                      'dog/N': 1,
                      'game/N': 2,
                      'kid/N': 3,
                      'like/V': 4,
                      'play/V': 5,
                      'cat/N_like/V': 6,
                      'kid/N_play/V': 7,
                      'like/V_dog/N': 8,
                      'play/V_game/N': 9}

    t.assert_array_equal(tr_matrix, training_matrix_signifier_bigrams)


def test_simple_bigram_features_with_chi2_fs():
    """
    A standard textbook setup with a limited number of useful bigram features, chi2 feature selection. No
    vector source needed.
    """
    tr_matrix, tr_voc, ev_matrix, ev_voc = _do_feature_selection(False, 5, vector_source=None,
                                                                 max_feature_len=2)
    assert ev_matrix.shape, (3, 5)

    # feature scores are [ 1.  1.  2.  2.  1.  2.  1.  2.  1.  2.]
    assert tr_voc == {  #'cat/N': 0, # removed because of low chi2-score
                        #'dog/N': 1,
                        'game/N': 0,
                        'kid/N': 1,
                        #'like/V': 4,
                        'play/V': 2,
                        #'cat/N like/V': 6,
                        'kid/N_play/V': 3,
                        #'like/V dog/N': 8,
                        'play/V_game/N': 4}
    t.assert_array_equal(tr_matrix, np.array([[0., 0., 0., 0., 0.],
                                              [0., 0., 0., 0., 0.],
                                              [1., 1., 1., 1., 1.]]))


def test_bigram_features_with_composer_without_fs():
    """
    A test with all uni- and bi-gram features and a simple predefined vector source for these bigrams. Feature
    handler is SignifierSignified to excercise nearest-neighbours look-up in the vector source
    """

    # load a mock unigram thesaurus, bypassing the similarity calculation provided by CompositeVectorSource
    composer = PrecomputedSimilaritiesVectorSource.from_file(
        thesaurus_files=['thesisgenerator/resources/exp0-0a.strings'])

    # patch it to ensure it contains some bigram entries, as if they were calculated on the fly
    composer.th['like/V_fruit/N'] = [('like/V', 0.8)]
    tr_matrix, tr_voc, ev_matrix, ev_voc = _do_feature_selection(False, 'all', handler='SignifierSignified',
                                                                 vector_source=composer, max_feature_len=2)
    assert ev_matrix.shape == (3, 10)
    t.assert_array_equal(tr_matrix, training_matrix_signifier_bigrams)

    # vector store says: fruit -> cat 0.06, like fruit -> like 0.8
    ev_expected = np.array([[0.06, 0., 0., 0., 1.8, 0., 0., 0., 0., 0.],
                            [0.06, 0., 0., 0., 1.8, 0., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]])

    t.assert_array_equal(ev_expected, ev_matrix)


def _check_debug_file(ev_matrix, tr_matrix, voc):
    for name, matrix in zip(['tr', 'ev'], [tr_matrix, ev_matrix]):
        filename = "PostVectDump_fs-test_%s-cl0-fold'NONE'.csv" % name
        df = read_csv(filename)
        # the columns are u'id', u'target', u'total_feat_weight', u'nonzero_feats', followed by feature vectors
        # check that we have the right number of columns
        assert len(df.columns) == 4 + len(voc)

        # check that column names match the vocabulary (after stripping feature metadata)
        #assertDictEqual(voc,
        #                     {' '.join(v.split('(')[1].split(')')[0].split(', ')): i
        #                      for i, v in enumerate(df.columns[4:])})

        # too much work to convert the strings back to a DocumentFeature object, just check the length
        assert len(voc) == len(df.columns[4:])
        #check that feature vectors are written correctly
        t.assert_array_equal(matrix, df.ix[:, 4:].as_matrix())
        os.remove(filename)

