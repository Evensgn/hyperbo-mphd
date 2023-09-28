"""
This codebase is based on the codebase of HyperBO. See more details in the README.md.

See more details of the HyperBO codebase in [`README.md`](https://github.com/google-research/hyperbo).
"""

from setuptools import find_packages
from setuptools import setup

setup(
    name='hyperbo',
    version='0.0.1',
    description='hyperbo-mphd',
    author='',
    author_email='',
    url='',
    license='Apache 2.0',
    packages=find_packages(),
    install_requires=[
        'orbax==0.1.0',
        'absl-py>=0.8.1',
        'clu',
        'flax<=0.7.0',
        'jax',
        'ml_collections',
        'numpy>=1.7',
        'optax',
        'pandas',
        'jaxopt==0.7',
        'tensorflow_probability',
        'tensorflow>=2.5.0',
        'xgboost',
        'pathos',
        'matplotlib',
    ],
    extras_require={},
    classifiers=[
        # https://pypi.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='bayesian optimization pre-training',
)
