import sys

sys.path.append('.')
from sklearn.cluster import KMeans
import numpy as np
from discoutils.thesaurus_loader import Vectors
import pandas as pd
from thesisgenerator.plugins.bov import ThesaurusVectorizer


def cluster_vectors(path_to_vectors, output_path, n_clusters=100, n_jobs=4):
    vectors = Vectors.from_tsv(path_to_vectors)
    km = KMeans(n_clusters=n_clusters, n_jobs=n_jobs, random_state=0)
    clusters = km.fit_predict(vectors.matrix)
    num2word = np.array(vectors.row_names)
    idx = np.argsort(num2word)
    df = pd.DataFrame(dict(clusters=clusters[idx]), index=num2word[idx])
    df.to_hdf(output_path, key='clusters', complevel=9, complib='zlib')


class KmeansVectorizer(ThesaurusVectorizer):
    def __init__(self, clusters_path, **kwargs):
        self.clusters_path = clusters_path
        super().__init__(**kwargs)

    def fit_transform(self, raw_documents, y=None):
        self.clusters = pd.read_hdf(self.clusters_path, key='clusters')
        return super().fit_transform(raw_documents, y=y)

    def _count_vocab(self, raw_documents, fixed_vocab):
        self.vocabulary_ = {'cluster%d' % i: i for i in range(len(self.clusters.clusters.unique()))}
        # vocabulary is fixed and equal to the number of topics, nothing to learn from text
        return super()._count_vocab(raw_documents, fixed_vocab=True)

    def _process_single_feature(self, feature, j_indices, values, *args):
        # insert cluster number of this features as its column number
        j_indices.append(self.clusters.ix[str(feature)][0])
        values.append(1)


if __name__ == '__main__':
    import sys

    infile, outfile, num_cl = sys.argv[1:]
    cluster_vectors(infile, outfile, int(num_cl))
