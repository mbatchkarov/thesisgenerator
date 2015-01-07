from pprint import pprint
from sklearn.svm import SVC
from thesisgenerator.plugins.bov import ThesaurusVectorizer
from thesisgenerator.utils.data_utils import load_text_data_into_memory, load_tokenizer, tokenize_data
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
import logging
import numpy as np

__author__ = 'mmb28'
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s\t%(module)s.%(funcName)s ""(line %(lineno)d)\t%(levelname)s : %(""message)s")

raw_data, data_ids = load_text_data_into_memory(
    training_path='sample-data/movie-reviews-train-tagged-full',
    test_path='sample-data/movie-reviews-test-tagged-full',
    shuffle_targets=False
)

tokenizer = load_tokenizer(
    joblib_caching=True,
    normalise_entities=False,
    use_pos=True,
    coarse_pos=True,
    lemmatize=True,
    lowercase=True,
    remove_stopwords=True,
    remove_short_words=False)
x_tr, y_tr, x_ev, y_ev = tokenize_data(raw_data, tokenizer, data_ids)

# for doc in x_ev:
#     for token in doc:
#         if ' ' in token:
#             print token

x_tr.extend(x_ev)
y = np.hstack((y_tr, y_ev))
vect = ThesaurusVectorizer(min_df=1, ngram_range=(0, 0),
                           extract_SVO_features=False, extract_VO_features=False)
data_matrix, voc = vect.fit_transform(x_tr, y)
with open('mr-ANs-NNs.txt', 'wb') as outf, open('mr-modifiers.txt', 'wb') as outf_mods:
    for item in voc.keys():
        if item.type in {'AN', 'NN'}:
            first = str(item.tokens[0])
            second = str(item.tokens[1])
            if item.type == 'AN':
                string = '{}:amod-HEAD:{}'.format(first, second)
            elif item.type == 'NN':
                string = '{}:nn-DEP:{}'.format(second, first)
            else:
                raise ValueError('There should only be AN and NN features')

            outf_mods.write('%s\n' % first)
            outf.write(string)
            outf.write('\n')


# # grid search for best SVM parameters
# print 'vectorizing'
# matrix, _ = vect.fit_transform(x_tr, y_tr)
# test_matrix = vect.transform(x_ev)

# parameters = {'C': [1, 3, 6, 10, 15, 20, 50, 100, 1000], 'gamma': [5, 2, 1, 0.1, 0.01, 0.001, 0.0001]}
# clf = SVC(kernel='rbf')
# grid_search = GridSearchCV(clf, parameters, n_jobs=-1, verbose=1,
#                            cv=KFold(matrix.shape[0], n_folds=6))
# print 'grid search'
# grid_search.fit(matrix, y_tr)
# print("Best score: %0.3f" % grid_search.best_score_)
# print("Best parameters set:")
# best_parameters = grid_search.best_estimator_.get_params()
# for param_name in sorted(parameters.keys()):
#     print("\t%s: %r" % (param_name, best_parameters[param_name]))
#
# pprint(sorted(grid_search.grid_scores_, key=lambda x: x.mean_validation_score))

