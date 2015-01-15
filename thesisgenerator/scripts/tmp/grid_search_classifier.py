import sys
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from thesisgenerator.plugins.bov import ThesaurusVectorizer
from thesisgenerator.utils.data_utils import load_text_data_into_memory, load_tokenizer, tokenize_data
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import logging
import numpy as np


def scorer(estimator, X, y):
    # return f1_score(y, estimator.predict(X), average='macro', pos_label=None)
    return accuracy_score(y, estimator.predict(X))


def go(tr_path, ev_path):
    raw_data, data_ids = load_text_data_into_memory(
        training_path=tr_path,
        test_path=ev_path
    )

    tokenizer = load_tokenizer(
        normalise_entities=False,
        use_pos=True,
        coarse_pos=True,
        lemmatize=True,
        lowercase=True,
        remove_stopwords=True,
        remove_short_words=False,
        remove_long_words=True)
    x_tr, y_tr, x_ev, y_ev = tokenize_data(raw_data, tokenizer, data_ids)

    x_tr.extend(x_ev)
    y = np.hstack((y_tr, y_ev))
    # x = x_tr
    # y = y_tr

    vect = ThesaurusVectorizer(min_df=1, ngram_range=(1, 1), unigram_feature_pos_tags=['N', 'J'],
                               extract_SVO_features=False, extract_VO_features=False, ngram_range_decode=(0, 0))
    # grid search for best SVM parameters
    logging.info('vectorizing')
    matrix, _ = vect.fit_transform(x_tr)

    parameters = {
        'clf__alpha': [np.power(10., i) for i in range(-5, 5)],
        # 'clf__gamma': [np.power(10., i) for i in range(-5, 5)]
    }
    logging.info('Parameters are %r', parameters)

    p = Pipeline([
        # ('scaler', StandardScaler()),
        ('clf', MultinomialNB())
    ])
    grid_search = GridSearchCV(p, parameters, n_jobs=1, verbose=1,
                               cv=StratifiedKFold(y, 10), scoring=scorer)
    logging.info('Starting grid search')
    grid_search.fit(matrix.A, y)
    logging.info("Best score: %0.3f" % grid_search.best_score_)
    logging.info("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        logging.info("\t%s: %r" % (param_name, best_parameters[param_name]))

    logging.info('All grid search scores:')
    all_scores = sorted(grid_search.grid_scores_, key=lambda x: x.mean_validation_score)
    logging.info('\n'.join(repr(x) for x in all_scores))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s ""(line %(lineno)d):\t %(""message)s")

    # go('sample-data/movie-reviews-train-tagged', 'sample-data/movie-reviews-test-tagged')
    # go('sample-data/reuters21578/r8train-tagged-grouped', 'sample-data/reuters21578/r8test-tagged-grouped')
    go('sample-data/reuters21578/r8train-tagged-grouped-med', 'sample-data/reuters21578/r8test-tagged-grouped-med')