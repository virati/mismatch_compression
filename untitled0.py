#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 19:52:19 2020

@author: virati
Building up a synthetic waveform
"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

t = np.linspace(0,10,1000)

x = np.zeros_like(t)
start = 100
x[start:start+20] = 1

plt.figure()
plt.subplot(211)
plt.plot(t,x)

f,Pxx = sig.welch(x=x,fs=1000/10,nfft=2**10,noverlap=200)
plt.subplot(212)
plt.plot(f,np.log10(np.abs(Pxx)))



#%%
# now we build up from Fourier series

y = np.zeros_like(t)
for nn in np.arange(0,100,2):
    y += 2/(nn+1) * np.sin(2 * np.pi * nn * t)
    
plt.figure()
plt.plot(y)