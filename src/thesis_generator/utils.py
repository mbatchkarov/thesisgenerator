# coding=utf-8

"""
A collection of random useful utilities
"""

def get_class(kls):
    """
    Imports a class by name. The call
        "my_import('my_package.my_module.my_class')"
    is equivalent to
        "from my_package.my_module import my_class"

    source Jason Baker: http://stackoverflow.com/questions/452969/does-python-have-an-equivalent-to-java-class-forname
    """
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    mod = __import__(module, fromlist=parts[-1])
    klass = getattr(mod, parts[-1])
    return klass

    #todo add function get_function(), which takes a fully quallified name and returns a function pointer