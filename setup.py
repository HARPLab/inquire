import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '1.0'
PACKAGE_NAME = 'inquire'
AUTHOR = 'Tesca Fitzgerald'
AUTHOR_EMAIL = 'tesca@cmu.edu'
URL = 'https://github.com/HARPLab/inquire'
LICENSE = 'BSD 3-Clause'
DESCRIPTION = 'INQUIRE: INteractive Querying for User-aware Informative REasoning'

INSTALL_REQUIRES = [
      'numpy',
      'scipy',
      'pygame'
]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages(exclude=['tests'])
      )