#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Setup script for Hybrid Quantum Emulator with Topological Compression"""

import io
import os
import re
import sys
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test as TestCommand
from codecs import open
from shutil import copyfile

# Package metadata
NAME = 'hybrid-quantum-emulator'
DESCRIPTION = 'Hybrid Quantum Emulator with Topological Compression and Photon-Inspired Architecture'
URL = 'https://github.com/quantum-research/hybrid-quantum-emulator'
EMAIL = 'info@quantum-emulator.org'
AUTHOR = 'Quantum Research Team'
REQUIRES_PYTHON = '>=3.8'
VERSION = '1.0.0a1'

# Required packages
REQUIRED = [
    'numpy>=1.21.0',
    'scipy>=1.7.0',
    'matplotlib>=3.4.0',
    'networkx>=2.6.0',
    'gudhi>=3.5.0',
    'ripser>=0.6.0',
    'scikit-learn>=1.0.0',
    'numba>=0.53.0',
    'tqdm>=4.62.0',
    'pandas>=1.3.0',
    'seaborn>=0.11.0',
    'persim>=0.2.0',
    'kmapper>=1.4.1',
    'tda-tools>=0.1.0',
    'qiskit>=0.32.0',
    'qiskit-aer>=0.9.0',
    'qiskit-ibmq-provider>=0.18.0',
    'cirq>=0.11.0',
    'pennylane>=0.19.0',
    'quantum-gates>=0.1.0'
]

# Optional packages
EXTRAS = {
    'gpu': [
        'cupy-cuda11x>=10.0.0; sys_platform == "linux"',
        'pycuda>=2021.1',
        'torch>=1.10.0',
        'torchvision>=0.11.0',
        'torchaudio>=0.10.0'
    ],
    'dev': [
        'pytest>=6.2.0',
        'pytest-cov>=3.0.0',
        'pytest-asyncio>=0.16.0',
        'hypothesis>=6.27.0',
        'mock>=4.0.0',
        'coverage>=6.0.0',
        'black>=21.9b0',
        'flake8>=3.9.0',
        'isort>=5.9.0',
        'pre-commit>=2.15.0',
        'jupyterlab>=3.2.0',
        'ipykernel>=6.4.0'
    ],
    'docs': [
        'sphinx>=4.2.0',
        'sphinx-rtd-theme>=1.0.0',
        'nbsphinx>=0.8.0',
        'm2r2>=0.3.0',
        'recommonmark>=0.7.0',
        'sphinxcontrib-napoleon>=0.7.0',
        'sphinx-autobuild>=2021.3.14'
    ],
    'windows': [
        'pywin32>=301'
    ],
    'macos': [
        'pyobjc-core>=8.0',
        'pyobjc-framework-Cocoa>=8.0'
    ]
}

# What packages are optional?
EXTRAS['all'] = list(set([item for sublist in EXTRAS.values() for item in sublist]))

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ''

    def run_tests(self):
        import shlex
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)

class BuildExt(build_ext):
    """Custom build_ext command to handle CUDA extensions."""
    
    def build_extension(self, ext):
        # Check if we're building CUDA extensions
        if ext.name.endswith('_cuda'):
            try:
                from torch.utils.cpp_extension import CUDAExtension
                # Replace with CUDAExtension if available
                ext = CUDAExtension(
                    name=ext.name,
                    sources=ext.sources,
                    include_dirs=ext.include_dirs,
                    library_dirs=ext.library_dirs,
                    libraries=ext.libraries,
                    language=ext.language,
                    extra_compile_args=ext.extra_compile_args
                )
            except ImportError:
                print("Warning: PyTorch not installed. Skipping CUDA extensions.")
                return
        
        build_ext.build_extension(self, ext)

# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    package_data={
        'hqe': [
            'platforms/*.yaml',
            'resources/*',
            'topology/gpu/*.cu',
            'core/platforms/*.yaml'
        ]
    },
    ext_modules=[
        Extension(
            'hqe.topology.gpu.ripser_cuda',
            ['src/topology/gpu/ripser_cuda.cpp'],
            include_dirs=['src/topology/gpu'],
            language='c++'
        )
    ],
    cmdclass={
        'build_ext': BuildExt,
        'test': PyTest,
    },
    entry_points={
        'console_scripts': [
            'hqe=hqe.cli:main',
            'hqe-cli=hqe.cli:main',
        ],
    },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Typing :: Typed'
    ],
    project_urls={
        'Documentation': 'https://hybrid-quantum-emulator.readthedocs.io/',
        'Source': 'https://github.com/quantum-research/hybrid-quantum-emulator',
        'Tracker': 'https://github.com/quantum-research/hybrid-quantum-emulator/issues',
        'Homepage': 'https://quantum-emulator.org'
    },
    keywords=[
        'quantum',
        'emulator',
        'topology',
        'quantum-computing',
        'quantum-algorithms',
        'photonic-computing'
    ],
    zip_safe=False,
    test_suite='tests',
    tests_require=EXTRAS['dev'],
    setup_requires=['setuptools>=58.0.0', 'wheel>=0.37.0', 'Cython>=0.29.0', 'numpy>=1.21.0']
)
