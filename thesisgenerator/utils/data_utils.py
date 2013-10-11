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
    # param keep_only_IT: the training data should not depend on the thesaurus, ie the keep_only_IT
    # intervention should only apply to decode time
    # param corpus_ids - list-like, names of the training corpus (and optional testing corpus), used for
    # retrieving pre-tokenized data from joblib cache
    x_tr, y_tr, x_test, y_test = data
    #todo this logic needs to be moved to feature extractor
    #tokenizer.keep_only_IT = False
    x_tr = tokenizer.tokenize_corpus(x_tr, corpus_ids[0])
    #tokenizer.keep_only_IT = keep_only_IT
    x_test = tokenizer.tokenize_corpus(x_test, corpus_ids[1])
    data = (x_tr, y_tr, x_test, y_test)
    return data


def load_text_data_into_memory(config):
    # read the raw text just once
    try:
        options = {'input': config['feature_extraction']['input'],
                   'shuffle_targets': config['shuffle_targets'],
                   'input_generator': config['feature_extraction']['input_generator']}
    except KeyError:
        # if the config dict is not created by configobj it may be missing some values
        # set these to some reasonable defaults
        options = {'input': 'content',
                   'shuffle_targets': False,
                   'input_generator': ''}
    options['source'] = config['training_data']
    options['test_data'] = config['test_data'] if config['test_data'] else None
    print 'Loading training data...'

    logging.info('Loading raw training set')
    x_train, y_train = _get_data_iterators(**options)
    if options['test_data']:
        logging.info('Loading raw test set')
        #  change where we read files from
        options['source'] = config['test_data']
        # ensure that only the training data targets are shuffled
        options['shuffle_targets'] = False
        x_test, y_test = _get_data_iterators(**options)
    return (x_train, y_train, x_test, y_test), (config['training_data'], config['test_data'])


def _load_tokenizer(config):
    """
    Initialises the state of helper modules from a config object
    """

    if config['joblib_caching']:
        memory = Memory(cachedir='.', verbose=0)
    else:
        memory = NoopTransformer()

    tok = tokenizers.XmlTokenizer(
        memory,
        normalise_entities=config['feature_extraction']['normalise_entities'],
        use_pos=config['feature_extraction']['use_pos'],
        coarse_pos=config['feature_extraction']['coarse_pos'],
        lemmatize=config['feature_extraction']['lemmatize'],
        lowercase=config['tokenizer']['lowercase'],
        remove_stopwords=config['tokenizer']['remove_stopwords'],
        remove_short_words=config['tokenizer']['remove_short_words'],
        use_cache=config['joblib_caching']
    )
    return tok


def _get_data_iterators(**kwargs):
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

    if kwargs['input'] == 'content' or kwargs['input'] == '':
        try:
            input_gen = kwargs['input_generator']
            source = kwargs['source']
            try:
                logging.debug(
                    'Retrieving input generator for name '
                    '\'%(input_gen)s\'' % locals())

                data_iterable = get_named_object(input_gen)(kwargs['source'])
                targets_iterable = np.asarray(
                    [t for t in data_iterable.targets()],
                    dtype=np.int)
                data_iterable = data_iterable.documents()
            except (ValueError, ImportError):
                logging.warn(
                    'No input generator found for name '
                    '\'%(input_gen)s\'. Using a file content '
                    'generator with source \'%(source)s\'' % locals())
                if not os.path.isdir(source):
                    raise ValueError('The provided source path (%s) has to be '
                                     'a directory containing data in the '
                                     'mallet '
                                     'format (class per directory, document '
                                     'per file). If you intended to load the '
                                     'contents of the file (%s) instead '
                                     'change '
                                     'the input type in main.conf to '
                                     '\'content\'')

                dataset = load_files(source, shuffle=False)
                logging.info('Targets are: %s' % dataset.target_names)
                data_iterable = dataset.data
                if kwargs['shuffle_targets']:
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
    elif kwargs['input'] == 'filename':
        raise NotImplementedError("The order of data and targets is wrong, "
                                  "do not use this keyword")
    elif kwargs['input'] == 'file':
        raise NotImplementedError(
            'The input type \'file\' is not supported yet.')
    else:
        raise NotImplementedError(
            'The input type \'%s\' is not supported yet.' % kwargs['input'])

    return data_iterable, targets_iterable