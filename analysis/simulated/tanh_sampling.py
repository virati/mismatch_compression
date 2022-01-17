#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 17:32:26 2020

@author: virati
tanh convolution of S(t) spectrum and sampling
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

def idxer(x,c):
    return np.argmin(np.abs(x-c))

tanh_expansions = [1,3,5,7] #we limit this to 5th since, empirically, I don't see the 7th order @ 90Hz; if you want more then do 7 and 9
shapings = [1,2,3,4,6]

w = np.linspace(-2000,2000,20000)
s = np.zeros((len(tanh_expansions),20000))

#s[0,idxer(w,105.5)] = 1
#s[0,idxer(w,-105.5)] = 1


for sidx,ss in enumerate(shapings):
    for idx,ii in enumerate(tanh_expansions):
        s[idx,idxer(w,ss*130*ii)] += 1
        s[idx,idxer(w,ss*-130*ii)] += 1



p = np.zeros_like(w)
p[idxer(w,0)] = 1
for nn in range(4):
    p[idxer(w,422*nn)] = 1
    p[idxer(w,-422*nn)] = 1

plt.figure()
plt.subplot(2,1,1)
for idx,ii in enumerate(tanh_expansions):
    
    plt.plot(w,s[idx,:],label=ii)
plt.plot(w,p)
plt.title('Stims and samples')
plt.legend()

plt.subplot(2,1,2)
#convolve
for idx,ii in enumerate((tanh_expansions)):
    res = sig.convolve(s[idx,:],p,mode='same')

    plt.plot(w,res,label=ii)
    
total_res = sig.convolve(np.sum(s,axis=0),p,mode='same')
plt.plot(w,-total_res,label='total',color='k')
plt.title('Convolved')
plt.xticks(ticks=np.arange(0,200,5))
plt.legend()
plt.xlim((-5,200))
