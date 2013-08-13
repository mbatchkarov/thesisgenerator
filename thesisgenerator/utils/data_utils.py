import logging
import os
from joblib import Memory
import numpy as np
from sklearn.datasets import load_files
from thesisgenerator.classifiers import NoopTransformer
from thesisgenerator.plugins import thesaurus_loader, tokenizers
from thesisgenerator.utils.conf_file_utils import parse_config_file

__author__ = 'mmb28'


def get_named_object(pathspec):
    """Return a named from a module.
    """
    logging.info('Getting named object %s' % pathspec)
    parts = pathspec.split('.')
    module = ".".join(parts[:-1])
    mod = __import__(module, fromlist=parts[-1])
    named_obj = getattr(mod, parts[-1])
    return named_obj


def tokenize_data(base_conf_file):
    config_obj, configspec_file = parse_config_file(base_conf_file)
    # read the raw text just once
    options = {'input': config_obj['feature_extraction']['input'],
               'shuffle_targets': config_obj['shuffle_targets']}
    try:
        options['input_generator'] = config_obj['feature_extraction'][
            'input_generator']
    except KeyError:
        options['input_generator'] = ''
    options['source'] = config_obj['training_data']
    if config_obj['test_data']:
        options['test_data'] = config_obj['test_data']
    print 'Loading training data...'
    x_tr, y_tr, x_test, y_test = load_text_data_into_memory(options)
    thesaurus, tokenizer = _init_utilities_state(config_obj)
    x_tr = map(tokenizer.tokenize, x_tr)
    x_test = map(tokenizer.tokenize, x_test)
    data = (x_tr, y_tr, x_test, y_test)
    return data, thesaurus


def load_text_data_into_memory(options):
    logging.info('Loading raw training set')
    x_train, y_train = _get_data_iterators(**options)
    if options['test_data']:
        logging.info('Loading raw test set')
        #  change where we read files from
        options['source'] = options['test_data']
        # ensure that only the training data targets are shuffled
        options['shuffle_targets'] = False
        x_test, y_test = _get_data_iterators(**options)
    return x_train, y_train, x_test, y_test


def _init_utilities_state(config):
    """
    Initialises the state of helper modules from a config object
    """

    if config['joblib_caching']:
        memory = Memory(cachedir='.', verbose=0)
    else:
        memory = NoopTransformer()

    th1 = thesaurus_loader.Thesaurus(
        config['feature_extraction']['train_thesaurus_files'],
        sim_threshold=config['feature_extraction']['sim_threshold'],
        include_self=config['feature_extraction']['include_self'])

    tok = tokenizers.XmlTokenizer(
        memory,
        normalise_entities=config['feature_extraction']['normalise_entities'],
        use_pos=config['feature_extraction']['use_pos'],
        coarse_pos=config['feature_extraction']['coarse_pos'],
        lemmatize=config['feature_extraction']['lemmatize'],
        thesaurus=th1,
        lowercase=config['tokenizer']['lowercase'],
        keep_only_IT=config['tokenizer']['keep_only_IT'],
        remove_stopwords=config['tokenizer']['remove_stopwords'],
        remove_short_words=config['tokenizer']['remove_short_words'],
        use_cache=config['joblib_caching']
    )
    return th1, tok


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