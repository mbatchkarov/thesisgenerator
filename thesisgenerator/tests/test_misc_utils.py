from thesisgenerator.utils.misc import (update_dict_according_to_mask, calculate_log_odds)
import numpy as np
from scipy.sparse import csr_matrix


def test_update_dict_according_to_mask():
    assert update_dict_according_to_mask({'a': 0, 'b': 1, 'c': 2}, [True, False, True]) == {'a': 0, 'c': 1}

    mask = np.array([True, False, True])
    assert update_dict_according_to_mask({'a': 0, 'b': 1, 'c': 2}, mask) == {'a': 0, 'c': 1}


def test_log_odds_score():
    X = [
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [0, 1, 1, 1],
        [0, 0, 1, 1],
    ]
    y = np.array([0, 0, 1, 1])
    res = calculate_log_odds(csr_matrix(X), y)
    print(res)
    assert res[0] > res[1] > res[2] > res[3]