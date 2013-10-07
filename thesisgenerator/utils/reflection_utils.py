import logging

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