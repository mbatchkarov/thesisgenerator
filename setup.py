# -*- coding: utf-8 -*-
from setuptools import setup

setup(
    name='thesisgenerator',
    version='0.4',
    packages=['thesisgenerator', 'thesisgenerator.plugins',
              'thesisgenerator.tests'],
    author=['Miroslav Batchkarov'],
    author_email=['M.Batchkarov@sussex.ac.uk'],
    install_requires=['iterpipes3', 'pandas', 'matplotlib', 'numpy', 'scipy',
                      'scikit-learn', 'joblib', 'configobj', 'pymysql',
                      'networkx', 'Cython', 'six', 'pandas'],
)

