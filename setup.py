# -*- coding: utf-8 -*-

from distutils.core import setup

setup(
    name='thesisgenerator',
    version='0.2',
    packages=['thesisgenerator', 'thesisgenerator.plugins',
              'thesisgenerator.tests'],
    author=['Matti Lyra', 'Miroslav Batchkarov'],
    author_email=['M.Lyra@sussex.ac.uk', 'M.Batchkarov@sussex.ac.uk'],
    requires=['iterpipes', 'pandas', 'matplotlib', 'numpy', 'scipy',
              'scikit-learn']
)

