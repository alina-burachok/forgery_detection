from distutils.core import setup, Extension

module1 = Extension('waveletsC', sources = ['waveletsC.c'])

setup (name = 'waveletsC',
        version = '1.0',
        description = 'oct2015',
        ext_modules = [module1])
