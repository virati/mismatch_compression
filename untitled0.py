#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 23:06:37 2020

@author: virati
Class for the 'stimulation waveform'
"""

import numpy as np
#import prettyplotlib as ppl
import matplotlib.pyplot as plt
import matplotlib
import scipy.signal as sig
import scipy.io as scio
import pandas as pd
import scipy

from collections import defaultdict

#import dill

from mpl_toolkits.mplot3d import Axes3D


font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

plt.rcParams['image.cmap'] = 'jet'

#%%


def gcd(a,b):
    while b>0:
        a,b = b,a % b
    return a

def lcm(a,b):
    return a*b / gcd(a,b)

def compute_GC_metric(F,Pxx):
    #compute the ratio of 64Hz to 130Hz to get "level of compression"
    F_harm2 = np.where(np.logical_and(F > 62, F < 64))
    F_harm1 = np.where(np.logical_and(F > 30, F < 34))

    #pow_ratio = 10*(np.log10(np.median(Pxx[F_harm2])) - np.log10(np.median(Pxx[F_harm1])))
    pow_ratio = (np.log10(np.median(Pxx[F_harm2]))/np.log10(np.median(Pxx[F_harm1])))
    #pow_ratio = np.median(Pxx[F_harm2])/np.median(Pxx[F_harm1])
    return {'Label':'Ratio of 2nd Harmonic and 1st Harmonic', 'Ratio':pow_ratio}


class stim_art:
    stim = 'ipg'
    Fs = 4220
    brFs = 422
    big_t = np.linspace(-10,10,Fs * 20)
    
    def __init__(self):
        ipg_infile = '/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/dLFP/ipg_data/ssipgwave_vreg_Ra1p1kOhm_1usdt.txt'
        inmatr = np.array(pd.read_csv(ipg_infile,sep=',',header=None))
        
        #concatenate this to massive
        concatstim = np.tile(inmatr[:,1],223)
        ccstim_tvect = np.linspace(0,concatstim.shape[0]/1e6,concatstim.shape[0]) - 10
        
        #now downsample this using interp
        artif = scipy.interpolate.interp1d(ccstim_tvect,concatstim)
        #save artif into a pickled function
            
        orig_big_x = artif(big_t)
        
        
        br_t = big_t[0::10] 
        
        #gaussian blur the orig_big_x to
        big_x = orig_big_x
        
        np.save('/tmp/ipg',orig_big_x)
        
        br_x = big_x[0::10]

    
    def do_sweep(self):
        g_sweep = np.linspace(0.01,5,500)
        #g_sweep = np.array([])
        
        measures = [None] * g_sweep.shape[0]
        
        for gg,g_fact in enumerate(g_sweep):
            big_x = np.tanh(g_fact*orig_big_x)
            #big_x = g_fact * orig_big_x
            
            #peak-to-peak of input voltage
            maxampl = np.max(np.abs(g_fact*big_x))
            maxoutampl = np.max(np.abs(big_x))
            
            #find the derivative of the tanh at the ptp val
            gc_level = (1 - np.tanh(maxampl)**2)
            br_x = big_x[0::10]
            F,Pxx = sig.welch(br_x,fs=brFs)
            
            
            print('Input max: ' + str(maxampl) + ' gives GC level: ' + str(gc_level))
            
            gc_measure = compute_GC_metric(F,Pxx)
            print(gc_measure)
            
            measures[gg] = {'MaxIn':maxampl,'MaxOut':maxoutampl,'gc_level':gc_level,'gc_metric':gc_measure}
            
    def plot_amplits(self):
        #plotting time!
        gc_fromg = np.array([rr['gc_metric']['Ratio'] for rr in measures])
        inampl_fromg = np.array([rr['MaxIn'] for rr in measures])
        outampl_fromg = np.array([rr['MaxOut'] for rr in measures])
        trueg_fromg = np.array([rr['gc_level'] for rr in measures])
        
        plt.figure()
        ax1 = plt.subplot(211)
        ax1.plot(g_sweep,inampl_fromg.T,label='Input Amplitude')
        #ax1.plot(g_sweep,outampl_fromg.T,label='Max Output Amplitude')
        ax1.plot(g_sweep,inampl_fromg.T - outampl_fromg.T,label='Amplitude Deficit')
        adef_perc = (inampl_fromg.T - outampl_fromg.T)/inampl_fromg.T
        #ax1.plot(g_sweep,adef_perc,label='Amplitude Deficit %')
        
        ax1.set_ylabel('Voltage')
        plt.legend()
        ax4 = ax1.twinx()
        ax4.plot(g_sweep,trueg_fromg.T,label='GC Truth',color='green')
        ax4.set_ylabel('Gain Compression',color='g')
        ax4.tick_params('y',colors='g')
        plt.legend()
        
        
        
        ax2 = plt.subplot(212)
        ax2.plot(g_sweep,gc_fromg.T,label='GC Measure',color='red')
        ax2.set_ylabel('GC Measure (ratio of harmonics)',color='red')
        ax2.tick_params('y',colors='r')
        ax3 = ax2.twinx()
        ax3.plot(g_sweep,trueg_fromg.T,label='GC Truth',color='green')
        ax3.set_ylabel('Gain Compression',color='g')
        ax3.tick_params('y',colors='g')
        plt.axvline(x=g_sweep[96])
        plt.legend()
        #%%
        plt.figure()
        plt.subplot(211)
        plt.plot(g_sweep,np.hstack((np.diff(adef_perc),np.array([0]))))
        plt.title('First Derivative of amplitude deficit between input and output')
        #find max
        idx_inflection = np.argmax(np.diff(adef_perc))
        plt.subplot(212)
        plt.plot(g_sweep,np.hstack((np.diff(np.diff(adef_perc)),np.array([0,0]))),label='Amplitude Deficit Deriv')
        plt.title('Second Derivative of amplitude deficit between input and output')
        
