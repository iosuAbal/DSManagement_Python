# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 12:29:23 2024

@author: Iosu Abal
"""

from setuptools import setup

setup(
   name='DSManagement',
   version='1.0.0',
   author='Iosu Abal',
   author_email='jabal002@ikasle.ehu.eus',
   packages=['DSManagement', 'DSManagement.test'],
   license='LICENSE.txt',
   description='Este paquete es un paquete para preprocesar bases de datos',
   long_description=open('README.txt').read(),
   tests_require=['pytest'],
   install_requires=[
      "seaborn >= 0.9.0", 
      "pandas >= 0.25.1", 
      "matplotlib >= 3.1.1", 
      "numpy >=1.17.2"
   ],
)