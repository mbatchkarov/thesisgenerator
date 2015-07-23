from discoutils.thesaurus_loader import Vectors
from collections import Counter
from discoutils.tokens import DocumentFeature


def info(path):
    print('------------', path)
    v = Vectors.from_tsv(path)
    print(Counter(DocumentFeature.from_string(x).type for x in v.keys()))
    print('total', len(v))

for c in 'Add Left Baroni Guevara Observed'.split():
    path = '/lustre/scratch/inf/mmb28/FeatureExtractionToolkit/exp11-13-composed-ngrams-ppmi-svd/AN_NN_wiki-100_%s.events.filtered.strings' % c
    info(path)

    path = '/lustre/scratch/inf/mmb28/FeatureExtractionToolkit/exp10-13-composed-ngrams-ppmi-svd/AN_NN_gigaw-100_%s.events.filtered.strings' % c
    info(path)

for c in 'Add Left'.split():
    path = '/lustre/scratch/inf/mmb28/FeatureExtractionToolkit/word2vec_vectors/composed/AN_NN_word2vec-gigaw_100percent-rep0_%s.events.filtered.strings' % c
    info(path)

    path = '/lustre/scratch/inf/mmb28/FeatureExtractionToolkit/word2vec_vectors/composed/AN_NN_word2vec-wiki_100percent-rep0_%s.events.filtered.strings' % c
    info(path)
