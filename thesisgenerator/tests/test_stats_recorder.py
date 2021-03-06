from collections import Counter
import pytest
import os
import pandas as pd
import numpy as np
from glob import glob
import six
from discoutils.thesaurus_loader import Thesaurus
from discoutils.tokens import DocumentFeature
from thesisgenerator.plugins.experimental_utils import run_experiment

"""
Run a full experiment with a simple dataset like
 Cats like dogs
 Kids play games
"""


def _get_counter_ignoring_negatives(df, column_list):
    c = Counter(np.ravel(df.ix[:, column_list].values))
    return Counter({k: v for k, v in c.items() if k >= 0})


@pytest.fixture(scope="module")
def stats_files(request):
    prefix = 'thesisgenerator/resources'
    # load a mock unigram thesaurus, bypassing the similarity calculation provided by CompositeVectorSource
    vector_source = Thesaurus.from_tsv('thesisgenerator/resources/exp0-0a.strings')
    # exp1 is like exp0, but using Signified encoding
    run_experiment(1, prefix=prefix, thesaurus=vector_source)

    # setup goes like this:
    # for each sample size K (here set to [3])
    #       repeat N times in parallel (crossvalidation)
    #           pick a sample of K training documents
    #           vectorize training and testing set
    #           for each classifier (here using MNB and BNB)
    #               do magic

    # This test is about the vectorization step, and that's the same across all K
    # iteration, because the same training data is used. We might as well just use the first result

    def fin():
        # remove the temp files produced by this test
        print('finalizing test')
        for f in glob('statistics/stats-tests-exp*'):
            print(f)
            os.unlink(f)

    request.addfinalizer(fin)

    # prefix of all stats files that get produced
    return ['statistics/stats-tests-exp1.tc.csv.gz', 'statistics/stats-tests-exp1.par.csv.gz']


def test_coverage_statistics(stats_files):
    """
    Check the number of features (unigram) in thesaurus and in vocabulary is right
    """
    full_df = pd.read_csv(stats_files[0], compression='gzip', comment='#', index_col=2)
    df = full_df[(full_df.stage == 'ev') & (full_df.cv_fold == 0)]  # decode time
    NUM_COLS = 5
    assert df.shape == (5, NUM_COLS)  # 5 types in the dataset
    assert df['count'].sum() == 9.  # tokens

    assert df.query('IV == 0 and IT == 0').shape == (2, NUM_COLS)  # 2 types both OOV and OOT
    assert df.query('IV == 0 and IT == 0')['count'].sum() == 4.0  # 4 tokens

    assert df.query('IV == 0 and IT > 0').shape == (1, NUM_COLS)
    assert df.query('IV == 0 and IT > 0')['count'].sum() == 2.0

    assert df.query('IV > 0 and IT == 0').shape == (0, NUM_COLS)
    assert df.query('IV > 0 and IT == 0')['count'].sum() == 0.0

    assert df.query('IV > 0 and IT > 0').shape == (2, NUM_COLS)
    assert df.query('IV > 0 and IT > 0')['count'].sum() == 3.0


    # at train time everything must be in vocabulary (that's how it works)
    # and in thesaurus (the test thesaurus is set up this way)
    df = full_df[(full_df.stage == 'tr') & (full_df.cv_fold == 0)]
    print(df)
    assert df.query('IV == 0 and IT == 0').shape == (0, NUM_COLS)  # 2 types both OOV and OOT
    assert df.query('IV == 0 and IT == 0')['count'].sum() == 0.0  # 4 tokens

    assert df.query('IV == 0 and IT > 0').shape == (0, NUM_COLS)
    assert df.query('IV == 0 and IT > 0')['count'].sum() == 0.0

    assert df.query('IV > 0 and IT > 0').shape == (6, NUM_COLS)
    assert df.query('IV > 0 and IT > 0')['count'].sum() == 9.0

    assert df.query('IV > 0 and IT == 0').shape == (0, NUM_COLS)
    assert df.query('IV > 0 and IT == 0')['count'].sum() == 0.0


def test_get_decode_time_paraphrase_statistics(stats_files):
    """
    Test the replacements made at decode time are right
    """

    # this test uses a signifier-signified encoding, i.e. only OOV-IT items are looked up
    df = pd.read_csv(stats_files[1], compression='gzip', comment='#', index_col=2).fillna(-1)
    NUM_COLS = 10
    assert df.shape == (6, NUM_COLS)

    assert _get_counter_ignoring_negatives(df, ['neigh%d_sim' % (i + 1) for i in range(3)]) == \
           Counter({.05: 2, .06: 2, .11: 2, .7: 2, .3: 2})  # 2 inserted items had a sim of .05, etc

    assert _get_counter_ignoring_negatives(df, ['available_replacements']) == \
           Counter({1: 2, 2: 4})  # 2 items have had 1 replacement, etc

    column_list = ['neigh%d' % (i + 1) for i in range(3)]
    # x>0 check filters out NaN-s
    types = [DocumentFeature.from_string(x).type for x in np.ravel(df.ix[:, column_list].values)
             if isinstance(x, six.string_types)]
    assert Counter(types) == Counter({'1-GRAM': 10})  # 10 total replacements, all of them unigrams
