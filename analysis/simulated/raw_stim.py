#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 00:22:21 2020

@author: virati
Raw stim waveform plot and PSD
"""
import numpy as np
#import prettyplotlib as ppl
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.io as scio
import pandas as pd

import sys
sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
from spot_check import spot_check

ipg_infile = '/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/dLFP/ipg_data/ssipgwave_vreg_Ra1p1kOhm_1usdt.txt'
inmatr = np.array(pd.read_csv(ipg_infile,sep=',',header=None))

def stim_plot(ts,Fs):
    plt.figure()
    plt.subplot(211)
    plt.plot(np.linspace(0,ts.shape[0]/Fs,ts.shape[0]),ts)
    
    plt.subplot(212)
    #pdb.set_trace()
    F,Pxx = sig.welch(ts,fs=Fs,nfft=2**15,nperseg=2**10,noverlap=2**10-5)
    plt.plot(F,(Pxx),linewidth=5,label='Sampled')
    #plt.xlim((0,500))
    return F

def raw():
    stim_plot(inmatr[:,1],Fs=1e6)
    
def decimated(factor=100):
    if factor == 1:
        ds_stim = inmatr[0:70000,1]    
    else:
        ds_stim = sig.decimate(inmatr[0:70000,1],q=factor)
        
    print(stim_plot(ds_stim,Fs=1e6/factor))

def empirical():
    fname = '/home/virati/MDD_Data/Benchtop/VRT_Impedance_RB/Session_2018_04_24_Tuesday/demo_2018_04_24_16_53_36__MR_0.txt'
    empirical_rec = spot_check(fname,tlims=(10,30),plot_sg=True)
    
decimated(factor=5)
#%%
#raw()
empirical()
