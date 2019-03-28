# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='earlywarningsignals',
    version='0.1.0',
    description='Toy package',
    long_description=readme,
    author='Our names here',
    author_email='our@gmail.com',
    url='https://github.com/PabRod/phdtools',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
