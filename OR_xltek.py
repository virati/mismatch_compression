#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:29:11 2020

@author: virati
OR - edf/XLTek file
"""
import mne
import matplotlib.pyplot as plt
import scipy.signal as sig
import numpy as np
#%%
file = "/home/virati/MDD_Data/OR/DBS901/DBS901.edf"
data = mne.io.read_raw_edf(file)
raw_data = data.get_data()
# you can get the metadata included in the file and a list of all channels:
info = data.info
channels = data.ch_names
fs=1000
#%%
s_start = 1000*60*35
s_len=60*10
chann=channels.index('SGC1')
plot_data = raw_data[chann,s_start:s_start+1000*s_len]
#%%
plt.figure()
plt.plot(sig.decimate(plot_data,10))




F,T,SG = sig.spectrogram(plot_data,fs = fs,nperseg=2**10,noverlap=512,nfft=2**11,window=('blackmanharris'))
#%%
plt.figure()
plt.pcolormesh(T,F,10*np.log10(SG),cmap='jet',rasterized=True)
plt.colorbar()
