#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 20:21:24 2018

@author: virati
Mess with IPG waveform
"""
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

import allantools
from allantools.noise import pink as f_noise

decay = 100
startfs = 1e6
endfs = 4220

# bvasic load of a 1/10th stim snippet
tenth_sec_stim = np.load(
    "/home/virati/Dropbox/projects/Research/MDD-DBS/Data/StimEphys/tenth_sec_ipg.npy"
)
# gaussian filter versus
tframe = (-10, 15)
tlen = tframe[1] - tframe[0]
full_stim = np.tile(tenth_sec_stim, 10 * 21)[0 : int(startfs * tlen)]
tvect = np.linspace(-10, 11, full_stim.shape[0])

#%%

conv_stim = full_stim

# do some bare-minimum filtering before the sampling
bl, al = sig.butter(2, 5e2 / startfs, btype="lowpass")
conv_stim = sig.lfilter(bl, al, conv_stim)

bh, ah = sig.butter(2, 10 / startfs, btype="highpass")
conv_stim = sig.lfilter(bh, ah, conv_stim)

# Now we're sampling the above convolved waveform
skip_ts = 238
meas_stim = conv_stim[0::skip_ts][0 : (endfs * 20)]
meas_stim[0 : endfs * 10] = 0
ds_tvect = np.linspace(-10, 10, meas_stim.shape[0])


#%%
nois_meas_stim = meas_stim

plt.figure()
plt.subplot(311)
plt.plot(ds_tvect, nois_meas_stim)

F, Pxx = sig.welch(nois_meas_stim[endfs * 10 :], fs=endfs)
plt.subplot(312)
plt.plot(F, 10 * np.log10(Pxx))

plt.subplot(313)
F, T, SG = sig.spectrogram(
    nois_meas_stim,
    nperseg=2 ** 8,
    noverlap=(2 ** 8) - 10,
    window=sig.get_window("blackmanharris", 2 ** 8),
    fs=endfs,
)
plt.pcolormesh(T, F, 10 * np.log10(SG))
