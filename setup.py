#!/usr/bin/env python

from setuptools import setup, find_packages
import os

# www.pythonhosted.org/setuptools/setuptools.html

exec(open('instamatic_stem/version.py').read())  # grab __version__, __author__, etc.

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

try:
    long_description = read('README.rst')
except IOError:
    long_description = read('README.md')

setup(
    name=__title__,
    version=__version__,
    description=__description__,
    long_description=long_description,

    author=__author__,
    author_email=__author_email__,
    url=__url__,

    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Microsoft :: Windows',
        'Topic :: Scientific/Engineering :: Human Machine Interfaces',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Software Development :: Libraries'
    ],

    packages=["instamatic_stem"],

    install_requires=['sounddevice', 
                      'numpy', 
                      'matplotlib'],

    package_data={
        "": ["LICENCE",  "readme.md", "setup.py"],
    },

    entry_points={
        'console_scripts': [
            # main
            'instamatic.stem       = instamatic_stem.gui.gui:main',
            'instamatic.stem.nogui = instamatic_stem.experiment:main',
        ]
    }
)

