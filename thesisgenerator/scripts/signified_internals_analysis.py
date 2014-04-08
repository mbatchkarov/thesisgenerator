from collections import Counter
import cPickle as pickle
from discoutils.tokens import DocumentFeature
import matplotlib.pyplot as plt
import pandas as pd
from thesisgenerator.plugins.stats import StatsRecorder, sum_up_token_counts
from collections import defaultdict


def histogram_from_list(l, path):
    MAX_LABEL_COUNT = 20

    plt.figure()
    if type(l[0]) == str:
        # numpy's histogram doesn't like strings
        s = pd.Series(Counter(l))
        s.plot(kind='bar', rot=0)
    else:
        plt.hist(l, bins=MAX_LABEL_COUNT)

    plt.savefig(path, format='png')


def get_basic_paraphrase_statistics(stats):
    '''
    Look at the kinds of replacements that are being made at decode time
    :param stats:
    :type stats: StatsRecorder
    :return:
    :rtype:
    '''
    replacement_count = [x.available_replacements for x in stats.paraphrases]
    replacement_rank = [r for x in stats.paraphrases for r in x.ranks]
    replacement_sims = [s for x in stats.paraphrases for s in x.similarities]
    replacement_types = [feat.type for x in stats.paraphrases for feat in x.replacements]
    return replacement_count, replacement_rank, replacement_sims, replacement_types


def get_classificational_similarity_statistics(stats):
    if hasattr(stats, 'ratio'):
        replacement_ratios = {}
        for p in stats.paraphras:
            if p.original in stats.ratio:  # decode-time feature seen in training
                original_value = stats.ratio[p.original]  # log class-conditional ratio
                replacement_value = ([stats.ratio[r] for r in p.replacements])
                print p, original_value, replacement_value

    return replacement_ratios


def do_work():
    fname = 'stats-exp0-0-cv0-tr'
    df = sum_up_token_counts(fname)
    vocab = set(df.index.tolist())
    print 'Vocabulary is %r' % list(vocab)
    df_it = df[df['IT'] > 0]
    print 'Train-time IT tokens: %d/%d' % (df_it['count'].sum(), df['count'].sum())
    print 'Train-time IT types: %d/%d' % (df_it.shape[0], df.shape[0])
    fname = 'stats-exp0-0-cv0-ev'
    df = sum_up_token_counts(fname)
    df_it = df[df['IT'] > 0]
    df_iv = df[df['IV'] > 0]
    print 'Decode-time IT tokens: %d/%d' % (df_it['count'].sum(), df['count'].sum())
    print 'Decode-time IT types: %d/%d' % (df_it.shape[0], df.shape[0])
    print 'Decode-time IV tokens: %d/%d' % (df_iv['count'].sum(), df['count'].sum())
    print 'Decode-time IV types: %d/%d' % (df_iv.shape[0], df.shape[0])
    print 'Decode-time IV IT tokens: %d' % (df[(df['IT'] > 0) & (df['IV'] > 0)]['count'].sum())
    print 'Decode-time OOV IT tokens: %d' % (df[(df['IT'] == 0) & (df['IV'] > 0)]['count'].sum())
    print 'Decode-time IV OOT tokens: %d' % (df[(df['IT'] > 0) & (df['IV'] == 0)]['count'].sum())
    print 'Decode-time OOV OOT tokens: %d' % (df[(df['IT'] == 0) & (df['IV'] == 0)]['count'].sum())
    df2 = pd.read_hdf(fname, 'paraphrases')
    df2.columns = ('feature', 'available_replacements', 'max_replacements',
                   'replacement1', 'replacement1_rank', 'replacement1_sim',
                   'replacement2', 'replacement2_rank', 'replacement2_sim',
                   'replacement3', 'replacement3_rank', 'replacement3_sim')
    counts = df2.groupby('feature').count().feature
    assert counts.sum() == df2.shape[0]  # no missing rows
    df2 = df2.drop_duplicates()
    df2.set_index('feature', inplace=True)
    df2['count'] = counts

    with open('%s.pickle' % fname) as infile:
        stats = pickle.load(infile)
        print 1
    # for i, stats in enumerate(stats_over_cv):
    #     data = stats.get_paraphrase_statistics()
    #     if not data:
    #         continue  # stats may be disabled for performance reasons
    #     for c, datatype in zip(data, ['count', 'rank', 'sim', 'feattype']):
    #         histogram_from_list(c,
    #                             'figures/%s_cv%d_%s_hist.png' % (fname, i, datatype))
    #
    try:
        flp = stats.nb_feature_log_prob
        # this only works for binary classifiers
        ratio = flp[0, :] - flp[1, :]  # P(f|c0) / P(f|c1) in log space
        # high positive value mean strong association with class 0, very negative means the opposite
        scores = {feature: ratio[index] for index, feature in stats.nb_inv_voc.items()}
        print scores
    except AttributeError:
        print 'Classifier parameters unavailable'
        return

    replacement_scores = defaultdict(int)
    for f in df2.index:
        try:
            orig_score = scores[DocumentFeature.from_string(f)]
        except KeyError:
            # this feature was not in the original vocabulary, we inserted it
            continue

        repl_count = df2.ix[f]['count']
        if repl_count > 0:
            for i in range(1, 4):
                replacement = df2.ix[f]['replacement%d' % i]
                repl_sim = df2.ix[f]['replacement%d_sim' % i] # they should all be the same
                if repl_sim > 0:
                    # -1 signifies no replacement has been found
                    repl_score = repl_sim * scores[DocumentFeature.from_string(replacement)]
                    replacement_scores[(orig_score, repl_score)] += repl_count
    x = []
    y = []
    thickness = []
    print replacement_scores
    for (orig_value, repl_value), repl_count in replacement_scores.iteritems():
        y.append(repl_value)
        x.append(orig_value)
        thickness.append(repl_count)
    print thickness
    plt.scatter(x, y, thickness)
    plt.hlines(0, min(x), max(x))
    plt.vlines(0, min(y), max(y))
    plt.xlabel('Class association of decode-time feature')
    plt.ylabel('Class association of replacements')
    plt.show()


if __name__ == '__main__':
    do_work()

