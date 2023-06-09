import sys
from distutils.core import setup
# from os.path import abspath, dirname, join
import setuptools

def version():
    version = {}
    with open("src/version.py") as fp:
        exec (fp.read(), version)
    return version['__version__']

__version__ = version()

setup(
    name='proes',
    version=__version__,
    author='Eloise Moore',
    author_email='22684463@student.uwa.edu.au',
    # url=''
    license='MIT',
    description='Gamma-ray burst PROmpt Emission Simulator',
    packages=['proes'],
    install_requires=[
        'numpy',
        'math',
        'matplotlib',
        'astropy',
        'random',
        'os',
        'path', 
        'scipy'
    ],
    classifiers=[
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    'Development Status :: 3 - Alpha',

    # Indicate who your project is intended for
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Physics',

    # Pick your license as you wish (should match "license" above)
    'License :: OSI Approved :: MIT License',

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    # 'Programming Language :: Python :: 3',
    # 'Programming Language :: Python :: 3.6',
    # 'Programming Language :: Python :: 3.7',
    # 'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.10',
],

)
