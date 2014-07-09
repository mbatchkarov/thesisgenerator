from collections import Counter
import pytest
import os
import pandas as pd
import numpy as np
from glob import glob
from discoutils.thesaurus_loader import Thesaurus
from discoutils.tokens import DocumentFeature
from thesisgenerator.plugins.stats import sum_up_token_counts
from thesisgenerator.plugins.experimental_utils import run_experiment


def _get_counter_ignoring_negatives(df, column_list):
    c = Counter(np.ravel(df.ix[:, column_list].values))
    return Counter({k: v for k, v in c.items() if k >= 0})


@pytest.fixture(scope="module")
def stats_file(request):
    prefix = 'thesisgenerator/resources'
    # load a mock unigram thesaurus, bypassing the similarity calculation provided by CompositeVectorSource
    vector_source = Thesaurus.from_tsv('thesisgenerator/resources/exp0-0a.strings')
    # exp1 is like exp0, but using Signified encoding
    run_experiment(1, num_workers=1, predefined_sized=[3],
                   prefix=prefix, thesaurus=vector_source)

    # setup goes like this:
    #   for each sample size K (here set to [3])
    #       repeat N times in parallel (crossvalidation)
    #           pick a sample of K training documents
    #           vectorize training and testing set
    #           for each classifier (here using MNB and BNB)
    #               do magic

    # This test is about the vectorization step, and that's the same across all K
    # iteration, because the same training data is used. We might as well just use the first result

    def fin():
        # remove the temp files produced by this test
        print 'finalizing test'
        for f in glob('statistics/stats-tests-exp*'):
            print f
            os.unlink(f)

    request.addfinalizer(fin)

    # prefix of all stats files that get produced
    return 'statistics/stats-tests-exp1-0-cv0-ev'


def test_coverage_statistics(stats_file):
    # decode time
    df = sum_up_token_counts('%s.tc.csv' % stats_file)
    assert df.shape == (5, 3)  # 5 types in the dataset
    assert df['count'].sum() == 9.  # tokens

    assert df.query('IV == 0 and IT == 0').shape == (2, 3)  # 2 types both OOV and OOT
    assert df.query('IV == 0 and IT == 0')['count'].sum() == 4.0  # 4 tokens

    assert df.query('IV == 0 and IT > 0').shape == (1, 3)
    assert df.query('IV == 0 and IT > 0')['count'].sum() == 2.0

    assert df.query('IV > 0 and IT == 0').shape == (0, 3)
    assert df.query('IV > 0 and IT == 0')['count'].sum() == 0.0

    assert df.query('IV > 0 and IT > 0').shape == (2, 3)
    assert df.query('IV > 0 and IT > 0')['count'].sum() == 3.0


    # at train time everything must be in vocabulary (that's how it works)
    # and in thesaurus (the test thesaurus is set up this way)
    df = sum_up_token_counts('%s.tc.csv' % stats_file.replace('-ev', '-tr'))

    assert df.query('IV == 0 and IT == 0').shape == (0, 3)  # 2 types both OOV and OOT
    assert df.query('IV == 0 and IT == 0')['count'].sum() == 0.0  # 4 tokens

    assert df.query('IV == 0 and IT > 0').shape == (0, 3)
    assert df.query('IV == 0 and IT > 0')['count'].sum() == 0.0

    assert df.query('IV > 0 and IT > 0').shape == (6, 3)
    assert df.query('IV > 0 and IT > 0')['count'].sum() == 9.0

    assert df.query('IV > 0 and IT == 0').shape == (0, 3)
    assert df.query('IV > 0 and IT == 0')['count'].sum() == 0.0


def test_get_decode_time_paraphrase_statistics(stats_file):
    """
    :param stats:
    :type stats: StatsRecorder
    """

    # this test uses a signifier-signified encoding, i.e. only OOV-IT items are looked up
    df = pd.read_csv('%s.par.csv' % stats_file, sep=', ')
    assert df.shape == (5, 12)

    assert _get_counter_ignoring_negatives(df, ['replacement%d_sim' % (i + 1) for i in range(3)]) == \
           Counter({.05: 2, .06: 2, .11: 2, .7: 1, .3: 1})  # 2 inserted items had a sim of .05, etc

    assert _get_counter_ignoring_negatives(df, ['replacement%d_rank' % (i + 1) for i in range(3)]) == \
           Counter({0: 5, 1: 3})  # 5 inserted items were the top neighbour, etc

    assert _get_counter_ignoring_negatives(df, ['available_replacements']) == \
           Counter({1: 2, 2: 3})  # 2 items have had 1 replacement, etc

    column_list = ['replacement%d' % (i + 1) for i in range(3)]
    # x>0 check filters out NaN-s
    types = [DocumentFeature.from_string(x).type for x in np.ravel(df.ix[:, column_list].values) if x > 0]
    assert Counter(types) == Counter({'1-GRAM': 8})  # 8 total replacements, all of them unigrams