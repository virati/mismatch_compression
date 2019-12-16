#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:03:10 2018

@author: virati
RIDGEd Inversion of tanh
"""

import scipy.signal as sig
import matplotlib.pyplot as plt
from diff_model import *
import numpy as np

#generate our true signal

def generator():
    sim_lfp = sig.detrend(brain_sig(1000,13,2e-7).ts_return(),type='constant')
    
    return sim_lfp


x_true = generator()

gain = 1

y_meas = gain * 5e-7 * np.tanh(x_true / 5e-7) + np.random.normal(0,2e-7,x_true.shape)

#%%


x_pred = 1e-6 * np.arctanh(y_meas / (np.max(np.abs(y_meas))+1e-9))

plt.plot(x_true,label='True')
plt.plot(y_meas,label='Measure')
plt.plot(x_pred,label='Predicted',alpha=0.5)
plt.legend()

# Plot the PSDs now
xs = [x_true, y_meas, x_pred]

plt.figure()
for xl,x in enumerate(xs):
    f,Pxx = sig.welch(x,fs=422)
    plt.plot(f,10*np.log10(Pxx),label=xl)
    
plt.legend()