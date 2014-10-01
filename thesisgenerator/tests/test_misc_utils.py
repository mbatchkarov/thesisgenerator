from thesisgenerator.utils.misc import update_dict_according_to_mask
import numpy as np


def test_update_dict_according_to_mask():
    assert update_dict_according_to_mask({'a': 0, 'b': 1, 'c': 2}, [True, False, True]) == {'a': 0, 'c': 1}

    mask = np.array([True, False, True])
    assert update_dict_according_to_mask({'a': 0, 'b': 1, 'c': 2}, mask) == {'a': 0, 'c': 1}