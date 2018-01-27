#!/usr/bin/env python 

from distutils.core import setup, Extension

setup(name='_makearq', version='0.1',
      description='make_arq helper',
      ext_modules=[Extension('_makearq', ['_makearq.c']
                             ## include_dirs=['/Developer/Headers/FlatCarbon'],
                             ## extra_compile_args=['-g'],
                             ## extra_link_args=['-g', '-framework', 'Carbon'],
                             )]
      )
