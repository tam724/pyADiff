from setuptools import setup, find_packages
import os, sys

version = {}
with open(os.path.join(sys.path[0], "pyADiff/version.py")) as f:
    exec(f.read(), version)

with open(os.path.join(sys.path[0], "README.md")) as f:
    readme = f.read()

setup(
    name='pyADiff',
    version=version['__version__'],
    author='Tamme Claus',
    author_email='tamme.claus@gmail.com',
    description='A simple, pure python algorithmic differentiation package',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/tam724/pyADiff',
    include_package_data=True,
    license="GNU GPLv3 ",
    packages=find_packages(exclude=('examples', 'tests', 'doc')),
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.17'
    ]
    )