from distutils.core import setup, Extension
import numpy

module1 = Extension('waveletsC', sources = ['waveletsC.c'], include_dirs=[numpy.get_include()])

setup (name = 'waveletsC',
        version = '1.0',
        description = 'oct2015',
        ext_modules = [module1])
