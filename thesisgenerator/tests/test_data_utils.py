import os
from thesisgenerator.utils.data_utils import (get_tokenized_data, jsonify_single_labelled_corpus,
                                              get_tokenizer_settings_from_conf)
from thesisgenerator.utils.conf_file_utils import parse_config_file
import networkx as nx
import numpy as np


def test_jsonify_XML_corpus():
    conf_file = 'thesisgenerator/resources/conf/exp0/exp0_base.conf'
    conf, _ = parse_config_file(conf_file)
    train_set = conf['training_data']
    test_set = conf['test_data']
    json_train_set = train_set + '.gz'
    tk = get_tokenizer_settings_from_conf(conf)

    # parse the XML directly
    x_tr, y_tr, x_ev, y_ev = get_tokenized_data(train_set, tk, gzip_json=False,
                                                test_data=test_set)

    jsonify_single_labelled_corpus(conf_file)
    x_tr1, y_tr1, _, _ = get_tokenized_data(json_train_set, tk, gzip_json=True)

    # because the process of converting to json merges the train and test set, if a test set exists,
    # we need to merge them too in this test.
    for a, b in zip(x_tr + x_ev, x_tr1):
        assert len(a) == len(b) == 1
        assert nx.is_isomorphic(a[0], b[0])
        # sanity check
        assert nx.is_isomorphic(b[0], b[0])
        assert nx.is_isomorphic(a[0], a[0])
    np.testing.assert_array_equal(np.hstack((y_tr, y_ev)), y_tr1)

    os.unlink(json_train_set)