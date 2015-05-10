import os
from thesisgenerator.utils.data_utils import (get_tokenized_data, jsonify_single_labelled_corpus,
                                              get_tokenizer_settings_from_conf)
from thesisgenerator.utils.conf_file_utils import parse_config_file
import numpy as np


def test_jsonify_XML_corpus():
    conf_file = 'thesisgenerator/resources/conf/exp0/exp0_base.conf'
    conf, _ = parse_config_file(conf_file)
    train_set = conf['training_data']
    json_train_set = train_set + '.gz'
    tk = get_tokenizer_settings_from_conf(conf)

    # parse the XML directly
    x_tr, y_tr, _, _ = get_tokenized_data(train_set, tk)

    jsonify_single_labelled_corpus(train_set)
    x_tr1, y_tr1, _, _ = get_tokenized_data(json_train_set, tk)

    # because the process of converting to json merges the train and test set, if a test set exists,
    # we need to merge them too in this test.
    for a, b in zip(x_tr, x_tr1):
        assert len(a[0]) == len(b) == 3
        assert set(str(f) for f in a[0].nodes()) == set(b)
    np.testing.assert_array_equal(y_tr, y_tr1)
    os.unlink(json_train_set)