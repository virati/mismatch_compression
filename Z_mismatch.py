#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 18:03:30 2019

@author: virati
File that performs the impedance mismatch analysis for patient set
"""

import DBSpace as dbo
from DBSpace.dLFP.impedances import Z_class
import matplotlib.pyplot as plt


print('Doing Impedance Mismatch Analysis - Script')
Z_lib = Z_class()
Z_lib.get_recZs()
Z_lib.plot_recZs()
Z_lib.dynamics_measures()