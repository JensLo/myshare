
from distutils.extension import Extension
from Cython.Build import cythonize
from distutils.core import setup
import numpy

name='mdfreader.dataRead'
version = '0.2.6'

description='A Measured Data Format file parser'


# What does your project relate to?
keywords='Parser MDF file'
# To provide executable scripts, use entry points in preference to the
# "scripts" keyword. Entry points provide cross-platform support and allow
# pip to create the appropriate form of executable for the target platform.


ext_modules=cythonize(Extension('dataRead', ['dataRead.pyx'], include_dirs=[numpy.get_include()], include_path=[numpy.get_include()]))

setup(name=name, version=version, description=description, keywords=keywords, ext_modules=ext_modules, include_path=[numpy.get_include()], include_dirs = [numpy.get_include()])

