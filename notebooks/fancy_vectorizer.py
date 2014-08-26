from collections import defaultdict
import array
import logging
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import f1_score, accuracy_score
from functools import partial
from functools32 import lru_cache
from operator import itemgetter
import gensim


class DistributionalVectorizer(CountVectorizer):
    """
    A simplified version of thesisgenerator's ThesaurusVectorizer for EuroScipy talk
    Assumes hybrid feature encoding, backed by a word2vec model

    """
    def __init__(self, model):
        self.thesaurus = model
        self.k = 3  # todo hardcoded value
        super(DistributionalVectorizer, self).__init__()  # todo not passing on any parameters

    def paraphrase(self, feature, vocabulary, j_indices, values, **kwargs):
        neighbours = self.get_nearest_neighbours(feature)
        neighbours = [foo for foo in enumerate(neighbours) if foo[0] in vocabulary]
        for neighbour, sim in neighbours[:self.k]:
            j_indices.append(vocabulary.get(neighbour))
            values.append(1.)  # todo not using similarity

    def insert_feature_only(self, feature_index_in_vocab, j_indices, values, **kwargs):
        j_indices.append(feature_index_in_vocab)
        values.append(1)

    def fit_transform(self, raw_documents, y=None):
        res = super(DistributionalVectorizer, self).fit_transform(raw_documents, y)
        self.init_sims(self.vocabulary_.keys())
        return res

    def _count_vocab(self, raw_documents, fixed_vocab):
        """
        Modified from sklearn 0.14's CountVectorizer

        @params fixed_vocab True if the vocabulary attribute has been set, i.e. the vectorizer is trained
        """
        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            # Add a new value when a new vocabulary item is seen, we're training now
            vocabulary = defaultdict(None)
            vocabulary.default_factory = vocabulary.__len__

        analyze = self.build_analyzer()
        j_indices = array.array(str("i"))
        indptr = array.array(str("i"))
        values = []
        indptr.append(0)
        for doc_id, doc in enumerate(raw_documents):
            for feature in analyze(doc):
                # ####################  begin non-original code  #####################

                try:
                    feature_index_in_vocab = vocabulary[feature]
                except KeyError:
                    feature_index_in_vocab = None
                    # if term is not in seen vocabulary

                is_in_vocabulary = feature in vocabulary
                is_in_th = feature in self.thesaurus if self.thesaurus else False

                # j_indices.append(feature_index_in_vocab) # todo this is the original code, also updates vocabulary

                params = {'doc_id': doc_id, 'feature': feature,
                          'feature_index_in_vocab': feature_index_in_vocab,
                          'vocabulary': vocabulary, 'j_indices': j_indices,
                          'values': values}
                if fixed_vocab:  # this object has been trained
                    if not is_in_vocabulary:
                        if is_in_th:
                            self.paraphrase(**params)
                        else:
                            pass  # nothing we can do
                    else:
                        self.insert_feature_only(**params)  # standard behaviour
                else:
                    self.insert_feature_only(**params)

            indptr.append(len(j_indices))

        if not fixed_vocab:
            # disable defaultdict behaviour
            vocabulary = dict(vocabulary)
            if not vocabulary:
                logging.error('Empty vocabulary')

        # some Python/Scipy versions won't accept an array.array:
        if j_indices:
            j_indices = np.frombuffer(j_indices, dtype=np.intc)
        else:
            j_indices = np.array([], dtype=np.int32)
        indptr = np.frombuffer(indptr, dtype=np.intc)

        X = sp.csr_matrix((values, j_indices, indptr),
                          shape=(len(indptr) - 1, len(vocabulary)),
                          dtype=self.dtype)
        X.sum_duplicates()  # nice that the summation is explicit
        return vocabulary, X

    def init_sims(self, vocab, n_neighbors=10):
        # self.name2row = {word: i for (i, word) in enumerate(self.thesaurus.index2word)}
        # selected_rows = [self.name2row[foo] for foo in vocab if foo in self.name2row]
        # if not selected_rows:
        # raise ValueError('None of the vocabulary items in the labelled set have associated vectors')
        # row2name = {v: k for k, v in self.name2row.items()}
        # self.selected_row2name = {new: row2name[old] for new, old in enumerate(selected_rows)}
        self.n_neighbours = n_neighbors

        self.selected_row2name = [f for f in vocab if f in self.thesaurus]
        matrix = np.vstack((self.thesaurus[f] for f in self.selected_row2name))
        num_vectors = matrix.shape[1]
        if n_neighbors > num_vectors:
            logging.warning('You requested %d neighbours to be returned, but there are only %d. Truncating.',
                            n_neighbors, num_vectors)
            n_neighbors = num_vectors - 1

        # BallTree/KDTree do not support cosine out of the box. algorithm='brute' may be slower overall
        # it's faster to build ,O(1), and slower to query
        self.nn = NearestNeighbors(algorithm='ball_tree',
                                   metric='euclidean',  # todo using euclidean instead of cosine with ball tree
                                   n_neighbors=n_neighbors + 1).fit(matrix)
        self.get_nearest_neighbours.cache_clear()

    @lru_cache(maxsize=2 ** 13)
    def get_nearest_neighbours(self, entry):
        """
        Get the nearest neighbours of `entry` amongst all entries that `init_sims` was called with. The top
        neighbour will never be the entry itself (to match Byblo's behaviour)
        """
        if entry not in self.thesaurus:
            raise ValueError('%s should have been in the models' % entry)

        distances, indices = self.nn.kneighbors(self.thesaurus[entry])
        neigh = [(self.selected_row2name[indices[0][i]], distances[0][i]) for i in range(indices.shape[1])]
        if neigh[0][0] == entry:
            neigh.pop(0)
        return neigh[:self.n_neighbours]


def eval_classifier(DATA_SIZES, NUM_CV, newsgroups_train, newsgroups_test, vect_callable=TfidfVectorizer):
    accuracy = np.zeros((len(DATA_SIZES), NUM_CV))
    for i, train_size in enumerate(DATA_SIZES):
        cv_iter = ShuffleSplit(len(newsgroups_train.data), n_iter=NUM_CV, train_size=train_size)
        for j, (train_idx, _) in enumerate(cv_iter):
            vectorizer = vect_callable()
            clf = MultinomialNB(alpha=.001)
            tr = vectorizer.fit_transform(itemgetter(*train_idx)(newsgroups_train.data))
            clf = clf.fit(tr, newsgroups_train.target[train_idx])
            ev = vectorizer.transform(newsgroups_test.data)
            score = accuracy_score(newsgroups_test.target, clf.predict(ev))
            accuracy[i, j] = score
    return accuracy


if __name__ == '__main__':
    model = gensim.models.Word2Vec.load('ass.word2vec.model')
    p = partial(DistributionalVectorizer, model)

    categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories,
                                          remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories,
                                         remove=('headers', 'footers', 'quotes'))
    DATA_SIZES = [10, 100, 500, 1000, len(newsgroups_test.data)]
    NUM_CV = 3

    print(eval_classifier(DATA_SIZES, NUM_CV, newsgroups_train, newsgroups_test, p))