from collections import Counter
import pytest
from thesisgenerator.composers.vectorstore import PrecomputedSimilaritiesVectorSource
from thesisgenerator.plugins.experimental_utils import run_experiment
import os
from glob import glob
from discoutils.tokens import DocumentFeature
import pandas as pd
import numpy as np
from thesisgenerator.plugins.stats import sum_up_token_counts


def _get_counter_ignoring_negatives(df, column_list):
    c = Counter(np.ravel(df.ix[:, column_list].values))
    return Counter({k: v for k, v in c.items() if k >= 0})


@pytest.fixture(scope="module")
def stats_file(request):
    prefix = 'thesisgenerator/resources'
    # load a mock unigram thesaurus, bypassing the similarity calculation provided by CompositeVectorSource
    vector_source = PrecomputedSimilaritiesVectorSource.from_file(
        thesaurus_files=['thesisgenerator/resources/exp0-0a.strings'])
    # exp1 is like exp0, but using Signified encoding
    results = run_experiment(1, num_workers=1, predefined_sized=[3],
                             prefix=prefix, vector_source=vector_source)
    # results is a list of result triples, one per training data size
    outfile, stats_objects = results[0]

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
        for f in glob('stats-tests-exp*'):
            print f
            os.unlink(f)

    request.addfinalizer(fin)
    return stats_objects[0].prefix


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
    df = pd.read_csv('%s.par.csv' % stats_file)
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