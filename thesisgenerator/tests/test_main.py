# coding=utf-8

from unittest import TestCase, skip

import numpy as np
import numpy.testing as t
import scipy.sparse as sp

from thesisgenerator.plugins import tokenizers, thesaurus_loader
from thesisgenerator import __main__


class Test_ThesaurusVectorizer(TestCase):
    def setUp(self):
        """
        Initialises the state of helper modules to sensible defaults
        """
        self.tokenizer = tokenizers.build_tokenizer(
            normalise_entities=False,
            use_pos=True,
            coarse_pos=True,
            lemmatize=True,
            lowercase=True,
            keep_only_IT=False)

        self._thesaurus_opts = {
            'thesaurus_files': ['thesisgenerator/resources/exp0-0a.strings'],
            'sim_threshold': 0,
            # 'k': 10,
            'include_self': False
        }

        self.feature_extraction_conf = {
            'vectorizer': 'thesisgenerator.plugins.bov.ThesaurusVectorizer',
            'use_tfidf': False,
            'min_df': 1,
            'lowercase': False,
            'replace_all': False
        }

        self.default_prefix = 'thesisgenerator/resources/test'

        self.data_options = {
            'input': 'content',
            'shuffle_targets': False,
            'input_generator': '',
        }

        self.x_tr, self.y_tr, self.x_ev, self.y_ev = self. \
            _load_data(self.default_prefix)

    def _load_data(self, prefix):
        """
        Loads a predefined dataset from disk
        """
        self.data_options['source'] = '%s-tr' % prefix
        x_tr, y_tr = __main__._get_data_iterators(**self.data_options)
        self.data_options['source'] = '%s-ev' % prefix
        x_ev, y_ev = __main__._get_data_iterators(**self.data_options)
        return x_tr, y_tr, x_ev, y_ev

    def test_get_data_iterators(self):
        """
        Tests if data loading functions returns the right number of documents
         and the right targets
        """
        x_tr, y_tr, x_ev, y_ev = self._load_data(self.default_prefix)
        # both training and testing data contain three docs
        self.assertEqual(len(x_tr), 3)
        self.assertEqual(len(x_ev), 3)
        self.assertEqual(len(y_ev), 3)
        self.assertEqual(len(y_tr), 3)

        # one doc of class 1 and two of class 0
        for y in [y_tr, y_ev]:
            self.assertEqual(y[0], 0)
            self.assertEqual(y[1], 0)
            self.assertEqual(y[2], 1)

    def _vectorize_data(self):
        pipeline = __main__._build_pipeline(
            12345, #id, for naming debug files
            None, # classifier
            self.feature_extraction_conf,
            {'run': False}, # feature selection conf
            {'run': False}, # dim re. conf
            None, # classifier options
            '.', # temp files dir
            True                # debug mode
        )
        x1 = pipeline.fit_transform(self.x_tr, self.y_tr)
        voc = pipeline.named_steps['vect'].vocabulary_
        x2 = pipeline.transform(self.x_ev)

        return x1, x2, voc

    def _reload_thesaurus(self):
        thesaurus_loader.read_thesaurus(**self._thesaurus_opts)

    @skip(
        "Not sure how the old algorithm below fits with our new thinking of"
        " what should happen when not using replace_all")
    def test_replaceAll_False_includeSelf_TrueFalse(self):
        self.feature_extraction_conf['replace_all'] = False

        for inc_self in [True, False]:
            # the expected matrices are the same with and without
            # include_self when replace_all=False
            self._thesaurus_opts['include_self'] = inc_self
            self._reload_thesaurus()

            x1, x2, voc = self._vectorize_data()

            # check vocabulary. For some reason it does not come out in the order
            #  in which words are put in, but that is fine as long as its the
            # same order every time
            self.assertDictEqual({'cat/n': 0, 'dog/n': 1, 'game/n': 2,
                                  'kid/n': 3, 'like/v': 4, 'play/v': 5},
                                 voc)

            # test output when not replacing all feature (old model)
            self.assertIsInstance(x1, sp.spmatrix)
            t.assert_array_equal(
                x1.todense(),
                np.array(
                    [
                        [1, 1, 0, 0, 1, 0],
                        [1, 1, 0, 0, 1, 0],
                        [0, 0, 1, 1, 0, 1]
                    ]
                )
            )

            t.assert_array_equal(
                x2.todense(),
                np.array(
                    [
                        [.06, .05, 0, 0, 1, 0],
                        [.06, .05, 0, 0, 1, 0],
                        [0, 1, 0, 0, 0, 0]
                    ]
                )
            )


            # ===============================================================
            # Test that the CSV dumper correctly unpacks the vocabulary and
            # the feature vectors
            # ===============================================================
            def compare_csv(expected, stage):
                expected = [x.strip() for x in expected.split()]
                filename = 'PostVectDump-%s12345.csv' % stage
                with open(filename) as infile:
                    csv_file_contents = [x.strip() for x in
                                         infile.readlines()]

                # headers must be identical character for character
                self.assertEqual(expected[0], csv_file_contents[0])
                for line1, line2 in zip(expected[1:],
                                        csv_file_contents[1:]):
                    for token1, token2 in zip(line1.split(','),
                                              line2.split(',')):
                        try:
                            print token1, token2
                            self.assertEqual(float(token1), float(token2))
                        except ValueError:
                            # in the evaluation file there are no targets,
                            # i.e. token1==''
                            self.assertEqual(token1, token2)
                            self.assertEqual(token1.strip(), '')

            expected = \
                """
    id,target,total_feat_weight,nonzero_feats,cat/n,dog/n,game/n,kid/n,like/v,play/v
    0,0,3,3,1,1,0,0,1,0
    1,0,3,3,1,1,0,0,1,0
    2,1,3,3,0,0,1,1,0,1
                """
            compare_csv(expected, 'tr')

            expected = \
                """
    id,target,total_feat_weight,nonzero_feats,cat/n,dog/n,game/n,kid/n,like/v,play/v
    0,,1.11,3,0.06,0.05,0,0,1,0
    1,,1.11,3,0.06,0.05,0,0,1,0
    2,,1,1,0,1,0,0,0,0
                """
            compare_csv(expected, 'ev')

    @skip("Do not use replace_all for now")
    def test_replaceAll_True_includeSelf_False(self):
        self.feature_extraction_conf['replace_all'] = True
        self._thesaurus_opts['include_self'] = False
        self._reload_thesaurus()

        x1, x2, voc = self._vectorize_data()

        self.assertDictEqual({'cat/n': 0, 'dog/n': 1, 'fruit/n': 2,
                              'game/n': 3, 'kid/n': 4, 'like/v': 5,
                              'play/v': 6},
                             voc)

        self.assertIsInstance(x1, sp.spmatrix)
        t.assert_array_equal(
            x1.toarray(),
            np.array(
                [
                    [.7, .8, 0, .4, 0.3, 0, 0.11],
                    [.7, .8, 0, .4, 0.3, 0, 0.11],
                    [.06, .04, 0.1, 0, 0, 0.09, 0.6]
                ]
            )
        )

        t.assert_array_equal(
            x2.toarray(),
            np.array(
                [
                    [.06, .05, 0, 0, 0, 0, 0.11],
                    [.06, .05, 0, 0, 0, 0, 0.11],
                    [.7, 0, 0, 0, 0.3, 0, 0]
                ]
            )
        )

    @skip("include_self makes no sense with replace_all at decode time. "
          "The nearest neighbour of each unknown token is going to be itself,"
          " which is OOV and will not be inserted")
    def test_replaceAll_True_includeSelf_True(self):
        self.feature_extraction_conf['replace_all'] = True
        self._thesaurus_opts['include_self'] = True
        self._reload_thesaurus()

        x1, x2, voc = self._vectorize_data()

        self.assertDictEqual({'cat/n': 0, 'dog/n': 1, 'fruit/n': 2,
                              'game/n': 3, 'kid/n': 4, 'like/v': 5,
                              'play/v': 6},
                             voc)

        self.assertIsInstance(x1, sp.spmatrix)
        t.assert_array_equal(
            x1.toarray(),
            np.array(
                [
                    [1.7, 1.8, 0, .4, 0.3, 1, 0.11],
                    [1.7, 1.8, 0, .4, 0.3, 1, 0.11],
                    [.06, .04, 0.1, 1, 1, 0.09, 1.6]
                ]
            )
        )

        t.assert_array_equal(
            x2.toarray(),
            np.array(
                [
                    [.06, .05, 1, 0, 0, 1, 0.11],
                    [.06, .05, 1, 0, 0, 1, 0.11],
                    [.7, 1, 0, 0, 0.3, 0, 0]
                ]
            )
        )

    def test_baseline_use_all_features_signifier_only_B(self):
        self.feature_extraction_conf['vocab_from_thes'] = False
        self.feature_extraction_conf['use_signifier_only'] = True
        self._thesaurus_opts['thesaurus_files'] = \
            ['thesisgenerator/resources/exp0-0b.strings']
        # self._thesaurus_opts['k'] = 1 # todo needs fixing
        self._reload_thesaurus()

        self.x_tr, self.y_tr, self.x_ev, self.y_ev = self. \
            _load_data('thesisgenerator/resources/test-baseline')

        x1, x2, voc = self._vectorize_data()

        self.assertDictEqual({'a/n': 0, 'b/n': 1, 'c/n': 2,
                              'd/n': 3, 'e/n': 4, 'f/n': 5},
                             voc)

        self.assertIsInstance(x1, sp.spmatrix)
        t.assert_array_equal(
            x1.toarray(),
            np.array(
                [
                    [1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1],
                ]
            )
        )

        t.assert_array_equal(
            x2.toarray(),
            np.array(
                [
                    [1, 0, 1, 0, 0, 0],
                ]
            )
        )

    def test_baseline_ignore_nonthesaurus_features_signifier_only_A(self):
        self.tokenizer.keep_only_IT = True
        self.feature_extraction_conf['use_signifier_only'] = True
        self._thesaurus_opts['thesaurus_files'] = \
            ['thesisgenerator/resources/exp0-0b.strings']
        # self._thesaurus_opts['k'] = 1
        self._reload_thesaurus()

        self.x_tr, self.y_tr, self.x_ev, self.y_ev = self. \
            _load_data('thesisgenerator/resources/test-baseline')

        x1, x2, voc = self._vectorize_data()

        self.assertDictEqual({'a/n': 0, 'b/n': 1, 'd/n': 2}, voc)

        self.assertIsInstance(x1, sp.spmatrix)
        t.assert_array_equal(
            x1.toarray(),
            np.array(
                [
                    [1, 1, 0],
                    [0, 0, 1],
                ]
            )
        )

        t.assert_array_equal(
            x2.toarray(),
            np.array(
                [
                    [1, 0, 0],
                ]
            )
        )

    def test_baseline_use_all_features_with_signified_D(self):
        self.tokenizer.keep_only_IT = False
        self.feature_extraction_conf['use_signifier_only'] = False
        self._thesaurus_opts['thesaurus_files'] = \
            ['thesisgenerator/resources/exp0-0b.strings']
        # self._thesaurus_opts['k'] = 1
        self._reload_thesaurus()

        self.x_tr, self.y_tr, self.x_ev, self.y_ev = self. \
            _load_data('thesisgenerator/resources/test-baseline')

        x1, x2, voc = self._vectorize_data()

        self.assertDictEqual({'a/n': 0, 'b/n': 1, 'c/n': 2,
                              'd/n': 3, 'e/n': 4, 'f/n': 5},
                             voc)

        self.assertIsInstance(x1, sp.spmatrix)
        t.assert_array_equal(
            x1.toarray(),
            np.array(
                [
                    [1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1],
                ]
            )
        )

        t.assert_array_equal(
            x2.toarray(),
            np.array(
                [
                    [1, 0, 1, 0.7, 0, 0],
                ]
            )
        )

    def test_baseline_ignore_nonthesaurus_features_with_signified_C(self):
        self.tokenizer.keep_only_IT = True
        self.feature_extraction_conf['use_signifier_only'] = False
        self._thesaurus_opts['thesaurus_files'] = \
            ['thesisgenerator/resources/exp0-0b.strings']
        # self._thesaurus_opts['k'] = 1 # equivalent to max
        self._reload_thesaurus()

        self.x_tr, self.y_tr, self.x_ev, self.y_ev = self. \
            _load_data('thesisgenerator/resources/test-baseline')

        x1, x2, voc = self._vectorize_data()

        self.assertDictEqual({'a/n': 0, 'b/n': 1, 'd/n': 2}, voc)

        self.assertIsInstance(x1, sp.spmatrix)
        t.assert_array_equal(
            x1.toarray(),
            np.array(
                [
                    [1, 1, 0],
                    [0, 0, 1],
                ]
            )
        )

        t.assert_array_equal(
            x2.toarray(),
            np.array(
                [
                    [1, 0, 0.7],
                ]
            )
        )
