from collections import Counter
import pytest
from thesisgenerator.composers.vectorstore import PrecomputedSimilaritiesVectorSource
from thesisgenerator.plugins.experimental_utils import run_experiment
import os
from glob import glob
from thesisgenerator.scripts.signified_internals_analysis import get_basic_paraphrase_statistics


@pytest.fixture(scope="module")
def stats(request):
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
        for f in glob('statstests-exp1-*'):
            print f
            os.unlink(f)

    request.addfinalizer(fin)
    return stats_objects[0]


def test_coverage_statistics(stats):
    assert False


def test_get_paraphrase_statistics(stats):
    """

    :param stats:
    :type stats: StatsRecorder
    """

    # this test uses a signifier-signified encoding, i.e. only OOV-IT items are looked up
    assert len(stats.paraphrases) == 5
    assert [Counter(x) for x in get_basic_paraphrase_statistics(stats)] == [
        Counter({1: 2, 2: 3}),  # 2 items have had 1 replacement, etc
        Counter({0: 5, 1: 3}),  # 5 inserted items were the top neighbour, etc
        Counter({.05: 2, .06: 2, .11: 2, .7: 1, .3: 1}),  # 2 inserted items had a sim of .05, etc
        Counter({'1-GRAM': 8})  # 8 total replacements, all of them unigrams
    ]