from collections import defaultdict
import array
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import scipy.sparse as sp


class DistributionalVectorizer(TfidfVectorizer):
    """
    A simplified version of thesisgenerator's ThesaurusVectorizer for EuroScipy talk
    Assumes hybrid feature encoding, backed by a word2vec model

    """
    # todo this class is painfully slow, see ./fancyvect.prof
    def __init__(self, model):
        self.thesaurus = model
        self.k = 1  # todo hardcoded value
        super().__init__()  # todo not passing on any parameters

    def paraphrase(self, feature, vocabulary, j_indices, values, **kwargs):
        neighbours = self.thesaurus.most_similar(feature)
        neighbours = [foo for foo in enumerate(neighbours) if foo[0] in vocabulary]

        for neighbour, sim in neighbours[:self.k]:
            j_indices.append(vocabulary.get(neighbour))
            values.append(sim)  # todo using raw similarity

    def insert_feature_only(self, feature_index_in_vocab, j_indices, values, **kwargs):
        j_indices.append(feature_index_in_vocab)
        values.append(1)

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


def eval_classifier(DATA_SIZES, NUM_CV, newsgroups_train, newsgroups_test, vect_callable=TfidfVectorizer):
    accuracy = np.zeros((len(DATA_SIZES), NUM_CV))
    for i, train_size in enumerate(DATA_SIZES):
        cv_iter = ShuffleSplit(len(newsgroups_train.data), n_iter=NUM_CV, train_size=train_size)
        for j, (train_idx, _) in enumerate(cv_iter):
            vectorizer = vect_callable()
            clf = MultinomialNB(alpha=.001)
            tr = vectorizer.fit_transform(np.array(newsgroups_train.data)[train_idx])
            clf = clf.fit(tr, newsgroups_train.target[train_idx])
            ev = vectorizer.transform(newsgroups_test.data[:10]) # todo remove slice
            score = accuracy_score(newsgroups_test.target[:10], clf.predict(ev)) # todo remove slice
            accuracy[i, j] = score
    return accuracy


if __name__ == '__main__':
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.cross_validation import ShuffleSplit
    from sklearn.metrics import f1_score, accuracy_score
    from functools import partial
    import gensim

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