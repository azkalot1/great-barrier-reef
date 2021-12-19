#!/usr/bin/env python

import re

from setuptools import setup, find_packages
from pathlib import Path


def read_requirements(req_path):
    """Read abstract requirements.
    Install requirements*.txt via pip first
    """
    # strip trailing whitespace, comments and URLs
    reqs = [re.sub(r'\s*([#@].*)?$', '', req) for req in Path(req_path).read_text().splitlines()]
    # skip empty lines
    return [req for req in reqs if req]


setup(
    name='great-barrier-reef',
    description='great-barrier-reef-kaggle',
    long_description=Path('README.md').read_text(),
    author='skolchenko',
    python_requires='>=3.8.6',
    install_requires=read_requirements('requirements.txt'),
    packages=find_packages(),
    zip_safe=True,
)