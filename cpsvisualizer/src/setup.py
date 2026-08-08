#!/usr/bin/env python
#coding:utf-8
import os
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

here = os.path.abspath(os.path.dirname(__file__))

try:
    README = open(os.path.join(here, 'README.md')).read()
except Exception:
    README = 'https://github.com/GeoPyTool/CPS-Visualizer/blob/main/README.md'

setup(
    name='cpsvisualizer',
    version='1.4.6',
    description='CPS-Visualizer: Visualization, similarity measurement, and statistical analysis of LA-ICP-MS surface scan data.',
    long_description=README,
    long_description_content_type='text/markdown',
    author='Qiu-Ye Yu, Ming-Hong Yao, Di Wang, Jian Cui, Zhi-Yan Wang, Fei-Fei Wang, Yan Liang',
    author_email='softcheck@outlook.com',
    url='https://github.com/GeoPyTool/CPS-Visualizer',
    packages=['cpsvisualizer'],
    package_data={
        'cpsvisualizer': ['*.py', '*.txt', '*.png', '*.qm', '*.ttf', '*.ini', '*.md'],
    },
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.26',
        'pandas',
        'scipy',
        'xlrd',
        'openpyxl',
        'matplotlib',
        'PySide6',
        'scikit-learn',
        'scikit-image',
        'joblib',
        'umap-learn',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Scientific/Engineering :: Chemistry',
    ],
)
