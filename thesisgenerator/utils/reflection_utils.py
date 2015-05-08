import inspect
import logging

__author__ = 'mmb28'


def get_named_object(pathspec):
    """Return a named from a module.
    """
    parts = pathspec.split('.')
    module = ".".join(parts[:-1])
    mod = __import__(module, fromlist=parts[-1])
    named_obj = getattr(mod, parts[-1])
    return named_obj


def get_intersection_of_parameters(klass, possible_param_values, prefix=''):
# the object must only take keyword arguments
    initialize_args = inspect.getargspec(klass.__init__)[0]
    if prefix:
        prefix += '__'
    return {'%s%s' % (prefix, arg): val for arg, val in possible_param_values.items() if arg in initialize_args}
