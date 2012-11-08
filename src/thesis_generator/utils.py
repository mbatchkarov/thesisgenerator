# coding=utf-8

"""
A collection of random useful utilities
"""

def get_named_object(pathspec):
    """Return a named from a module.
    """
    parts = pathspec.split('.')
    module = ".".join(parts[:-1])
    mod = __import__(module)
    named_obj = getattr(mod, parts[-1])
    return named_obj

    #todo add function get_function(), which takes a fully quallified name and returns a function pointer