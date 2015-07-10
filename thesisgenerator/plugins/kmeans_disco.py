import logging
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
    def fit_transform(self, raw_documents, y=None, clusters=None, **kwargs):
        if clusters is None:
            raise ValueError('Need a clusters file to fit this model')
        self.clusters = clusters
        return super().fit_transform(raw_documents, y=y)

    def _process_single_feature(self, feature, j_indices, values, vocabulary):
        try:
            # insert cluster number of this features as its column number
            cluster_id = self.clusters.ix[str(feature)][0]
            feature_index_in_vocab = vocabulary['cluster%d' % cluster_id]
            j_indices.append(feature_index_in_vocab)
            values.append(1)
        except KeyError:
            # the feature is not contained in the distributional model, ignore it
            pass


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %(message)s",
                        datefmt='%m-%d %H:%M')
    logging.info('Starting clustering')
    infile, outfile, num_cl = sys.argv[1:]
    cluster_vectors(infile, outfile, int(num_cl))
    logging.info('Done')
