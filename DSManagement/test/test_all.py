# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 12:36:13 2024

@author: Iosu Abal
"""
from DSManagement import dataset as dt
from DSManagement import plots as pp
from DSManagement import myUtils as ut
import numpy as np

def test_all():
    """Executes the tests to verify the installation has ben successfull"""
    ut.variance_base([1,2,3,4,5])
    at1 = dt.Atributo('altura',np.random.uniform(100, 200, 100))
    at2 = dt.Atributo('Esta_enfermo',np.random.choice([True, False], size=100))
    at3 = dt.Atributo('peso',np.random.uniform(50, 100, 100))
    at4 = dt.Atributo('Tiene_gripe',np.random.choice(['si', 'no'], size=100))
    at5 = dt.Atributo('edad', np.random.randint(0, 101, 100))
    ds = dt.Dataset("Pacientes",[at1,at2,at3,at4,at5])
    
    at1.discretize_EF(2)
    ds.discretize_EW(2)
    
    ds.calculate_metrics()
    
    ds.standarize().print()
    at1.normalize().print()
    
    ds.filter_metrics('Variance','>',300).print()
    
    ds.correlation()
    
    pp.plot_roc(ds, 'peso')
    pp.plot_correlation(ds)

