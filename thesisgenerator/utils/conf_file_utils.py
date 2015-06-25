from os import path as path
import os
from pprint import pprint
import sys
from configobj import ConfigObj, flatten_errors
import validate

__author__ = 'mmb28'


def set_nested(dic, key_list, value):
    """
    >>> d = {}
    >>> nested_set(d, ['person', 'address', 'city'], 'New York')
    >>> d
    {'person': {'address': {'city': 'New York'}}}
    """
    for key in key_list[:-1]:
        dic = dic.setdefault(key, {})
    dic[key_list[-1]] = value


def set_in_conf_file(conf_file, keys, new_value):
    if type(keys) is str:
        # handle the case when there is a single key
        keys = [keys]

    config_obj, configspec_file = parse_config_file(conf_file)
    set_nested(config_obj, keys, new_value)
    config_obj.write()


def parse_config_file(conf_file):
    if not os.path.exists(conf_file):
        raise ValueError('Conf file %s does not exits!'%conf_file)
    configspec_file = get_confrc(conf_file)
    config = ConfigObj(conf_file, configspec=configspec_file)
    validator = validate.Validator()
    result = config.validate(validator, preserve_errors=True)
    if result != True and len(result) > 0:
        print('Invalid configuration')
        pprint(flatten_errors(config, result))
        sys.exit(1)
    return config, configspec_file


def get_confrc(conf_file):
    """
    Searches the file hierarchy top to bottom for confrc,
    starting from conf_file and going as many as 4 levels up. As it goes
    up, also searches in the 'conf' sub directory, it exists
    """

    for subdir in ['.', 'conf']:
        for i in range(7):
            my_list = [path.dirname(conf_file)] + ['..'] * i + \
                      [subdir] + ['confrc']
            candidate = path.join(*my_list)
            if path.exists(candidate):
                return candidate