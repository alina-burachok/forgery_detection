from distutils.core import setup, Extension
import numpy

module1 = Extension('waveletsC2', sources = ['waveletsC2.c'], include_dirs=[numpy.get_include()])

setup (name = 'waveletsC2',
        version = '1.0',
        description = 'jan2018',
        ext_modules = [module1])
