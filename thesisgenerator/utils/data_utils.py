import logging
import os
from joblib import Memory
import numpy as np
from sklearn.datasets import load_files
from thesisgenerator.classifiers import NoopTransformer
from thesisgenerator.plugins import tokenizers
from thesisgenerator.utils.reflection_utils import get_named_object

__author__ = 'mmb28'


def tokenize_data(data, tokenizer, corpus_ids):
    # param corpus_ids - list-like, names of the training corpus (and optional testing corpus), used for
    # retrieving pre-tokenized data from joblib cache
    x_tr, y_tr, x_test, y_test = data
    #todo this logic needs to be moved to feature extractor
    x_tr = tokenizer.tokenize_corpus(x_tr, corpus_ids[0])
    x_test = tokenizer.tokenize_corpus(x_test, corpus_ids[1])
    data = (x_tr, y_tr, x_test, y_test)
    return data


def load_text_data_into_memory(training_path, test_path=None, input_generator='', shuffle_targets=False):
    # read the raw text just once
    print 'Loading training data...'

    logging.info('Loading raw training set')
    x_train, y_train = _get_data_iterators(training_path, input_gen=input_generator,
                                           shuffle_targets=shuffle_targets)

    if test_path:
        x_test, y_test = _get_data_iterators(training_path, input_gen=input_generator,
                                             shuffle_targets=shuffle_targets)
    return (x_train, y_train, x_test, y_test), (training_path, test_path)


def _load_tokenizer(normalise_entities, use_pos, coarse_pos, lemmatize,
                    lowercase, remove_stopwords, remove_short_words, joblib_caching):
    """
    Initialises the state of helper modules from a config object
    """

    if joblib_caching:
        memory = Memory(cachedir='.', verbose=0)
    else:
        memory = NoopTransformer()

    tok = tokenizers.XmlTokenizer(
        memory,
        normalise_entities=normalise_entities,
        use_pos=use_pos,
        coarse_pos=coarse_pos,
        lemmatize=lemmatize,
        lowercase=lowercase,
        remove_stopwords=remove_stopwords,
        remove_short_words=remove_short_words,
        use_cache=joblib_caching
    )
    return tok


def _get_data_iterators(path, input_type='content', input_gen=None, shuffle_targets=False):
    """
    Returns iterators over the text of the data.

    The *input_generator* option in the main configuration file is an
    optional argument for *feature_extract*. It should specify
    the fully qualified name of a generator class with two
    methods *documents* and *classes*. If the vectorizer's
    *input* value 'content' the *input_generator* will be used to
    feed the raw documents to the vectorizer.

    If the *input_generator* is not defined and the *input* field is
    *content* the source folder specified on the command line will be used
    as the input.  The source folder should in this case contain data in the
     mallet format. The same applies if the value of *input* is *filename*.

    See the documentation of the CountVectorizer in
    sklearn.feature_extraction.text for details on the parameter values.
    """

    def _filename_generator(file_list):
        for f in file_list:
            yield f

    def _content_generator(file_list):
        for f in file_list:
            with open(f, 'rb') as fh:
                yield fh.read()

    if input_type == 'content' or input_type == '':
        try:
            input_gen = input_gen
            #source = kwargs['source']
            try:
                logging.debug(
                    'Retrieving input generator for name '
                    '\'%(input_gen)s\'' % locals())
                if not input_gen:
                    raise ImportError
                data_iterable = get_named_object(input_gen)(path)
                targets_iterable = np.asarray(
                    [t for t in data_iterable.targets()],
                    dtype=np.int)
                data_iterable = data_iterable.documents()
            except (ValueError, ImportError):
                logging.warn(
                    'No input generator found for name '
                    '\'%(input_gen)s\'. Using a file content '
                    'generator with source \'%(path)s\'' % locals())
                if not os.path.isdir(path):
                    raise ValueError('The provided source path (%s) has to be '
                                     'a directory containing data in the '
                                     'mallet '
                                     'format (class per directory, document '
                                     'per file). If you intended to load the '
                                     'contents of the file (%s) instead '
                                     'change '
                                     'the input type in main.conf to '
                                     '\'content\'')

                dataset = load_files(path, shuffle=False)
                logging.info('Targets are: %s' % dataset.target_names)
                data_iterable = dataset.data
                if shuffle_targets:
                    import random

                    logging.warn('RANDOMIZING TARGETS')
                    random.shuffle(dataset.target)
                targets_iterable = dataset.target
        except KeyError:
            raise ValueError('Can not find a name for an input generator. '
                             'When the input type for feature extraction is '
                             'defined as content, an input_generator must '
                             'also '
                             'be defined. The defined input_generator should '
                             'produce raw documents.')
    elif input_type == 'filename':
        raise NotImplementedError("The order of data and targets is wrong, "
                                  "do not use this keyword")
    elif input_type == 'file':
        raise NotImplementedError(
            'The input type \'file\' is not supported yet.')
    else:
        raise NotImplementedError(
            'The input type \'%s\' is not supported yet.' % input_type)

    return data_iterable, targets_iterable