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
    author='Ingrid van de Leemput, M Usman Mirza, Pablo Rodriguez-Sanchez',
    author_email='ingrid.vandeleemput@wur.nl, muh.usmanmirza@gmail.com, pablo.rodriguez.sanchez@gmail.com',
    url='https://github.com/gossips/earlywarningsignals.git',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'data', 'R', 'vignettes'))
)
