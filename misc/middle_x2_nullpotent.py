#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 17:40:59 2018

@author: virati
Zero potential for x_2
This gives the equipotential line for when a source will be zerod out by differential channel
"""

import numpy as np
import matplotlib.pyplot as plt

class sig2:
    def __init__(self):
        self.compute_null()
       
        
    def compute_null(self,Z1=1000,Z3=1200,Zb=1e3):
        #c12 = np.linspace(3,10,100)
        #c23 = np.linspace(3,10,100)
        
        p1 = np.linspace(-10,10,1000)
        p2 = np.linspace(-6,6,1000)
        
        P1,P2 = np.meshgrid(p1,p2)
        
        #The factor; a free parameter to accentuate curve based on internals of PC+S that are unknown
        af=10
        
        c23 = af*np.sqrt(P1**2 + (3-P2)**2)
        c12 = af*np.sqrt(P1**2 + (-3-P2)**2)
        
        #c23,c12 = np.meshgrid(C3,C1)
        
        #potent = (Z1*c12**2 - Z3*c23**2)/(Zb*c12*c23) + (c23**2 - c12**2)/(c12*c23) - (Z1-Z3)/Zb
        potent = c12*Z1 - c23*Z3 - Zb*(c23-c12)
        
        self.potent = potent
        plt.figure()
        #h = plt.imshow(np.log10(np.abs(potent)),vmin=0,vmax=1,origin='lower')
        h = plt.imshow((np.abs(potent)),origin='lower',extent=[-10,10,-6,6])
        plt.colorbar()
        #plt.figure()
        cset = plt.contour(p1,p2,(potent),levels=[0],linewidths=2,colors='red')
        plt.title('af = ' + str(af))
        
        
        #eta = 15
        #less_locs = np.where(np.abs(potent < eta))

zero_field = sig2()