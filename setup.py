# -*- coding: utf-8 -*-

import os
from setuptools import setup
from maptiles import __version__

readmefile = os.path.join(os.path.dirname(__file__), "README.md")
with open(readmefile) as f:
    readme = f.read()

setup(
    name='maptiles',
    version=__version__,
    description='Create map images and use as plot background',
    author='Kota Mori', 
    author_email='kmori05@gmail.com',
    long_description=readme,
    long_description_content_type='text/markdown',
    url='https://github.com/kota7/maptiles',
    
    packages=['maptiles'],
    #py_modules=[],
    install_requires=['requests', 'pillow', 'numpy', 'matplotlib', 'pyproj']
)
