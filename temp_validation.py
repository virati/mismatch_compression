#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 13:30:57 2020

@author: virati
Temperature dependence code
"""

import sys
sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
from spot_check import spot_check
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import DBSpace as dbo
from DBSpace import nestdict

#%%
v_files = {'C36':'/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/mismatch_compression/data/temp_sweep/pt988_2017_02_19_11_53_52__MR_0_36C.txt',
           'C37':'/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/mismatch_compression/data/temp_sweep/pt988_2017_02_19_11_29_41__MR_0_37C.txt',
           'C38':'/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/mismatch_compression/data/temp_sweep/pt988_2017_02_19_12_17_01__MR_0_38C.txt'}


band_approach = 'Adjusted'
do_calc = 'mean'

do_side = 'Left'

#%%

def normalize(x):
    return x / (np.max(x)+1)

chann_label = ['Left','Right']
side_idx = {'Left':0,'Right':1}

stim_vs = {0:(0,10),5:(50,100)} #the timeperiods being studied as the 'voltage' times
do_stimvs = [0,5]

exp_results = nestdict()


preproc_flows = [['Pxx','Osc']]#,['Pxx_corr','Osc_corr']]
do_feats = {'Standard':['Delta','Theta','Alpha','Beta','Gamma'],'Adjusted':['Delta','Theta','Alpha','Beta*','Gamma1']}

for gel,fname in v_files.items():
    _ = spot_check(fname,tlims=(0,-1),plot_sg=True,chann_labels=[do_side])
    
    # Go to each voltage in the sweep
    for stim_v,iv in stim_vs.items():
        exp_results[stim_v] = spot_check(fname,tlims=iv,plot_sg=False)
        plt.suptitle(gel)
        
        #try some timedomain corrections
        #precorr_td = {chann:normalize(exp_results[stim_v]['TS'][:,cc]) for cc,chann in enumerate(chann_label)}
        #exp_results[stim_v]['TS_Corr'] = np.array([np.arctanh(precorr_td[chan]) for chan in chann_label])
        
        #Get ready for CORRECTION
        precorr_psd = {cc:10*(exp_results[stim_v]['F']['Pxx'][cc]) for cc in chann_label}
        
        poly_ret = dbo.poly_subtrEEG(precorr_psd,exp_results[stim_v]['F']['F'])
        
        exp_results[stim_v]['F']['Pxx_corr'] = poly_ret[0]
        exp_results[stim_v]['F']['PolyItself'] = poly_ret[1]
        
        
        #now do oscillatory state stuff
        #Uncorrected
        state_vect,state_basis = dbo.calc_feats(exp_results[stim_v]['F']['Pxx'],exp_results[stim_v]['F']['F'],modality='lfp',dofeats=do_feats[band_approach],compute_method=do_calc)
        exp_results[stim_v]['Osc'] = {'State':state_vect,'Basis':state_basis}
        
        state_vect,state_basis = dbo.calc_feats(exp_results[stim_v]['F']['Pxx_corr'],exp_results[stim_v]['F']['F'],modality='lfp',dofeats=do_feats[band_approach],compute_method=do_calc)
        exp_results[stim_v]['Osc_corr'] = {'State':state_vect,'Basis':state_basis}
        
        #Finally, we're just going to do the Gain Compression Ratio
        exp_results[stim_v]['GCratio'],_ = dbo.calc_feats(exp_results[stim_v]['F']['Pxx'],exp_results[stim_v]['F']['F'],modality='lfp',dofeats=['GCratio'],compute_method=do_calc)


    #go to each preprocessing flow
    for plot_proc in preproc_flows:
        plt.figure()
        for v_stim in do_stimvs:
            plt.subplot(2,1,1)
            plt.plot(exp_results[v_stim]['F']['F'],10*np.log10(exp_results[v_stim]['F'][plot_proc[0]][do_side]),linewidth=3)
            plt.xlim((0,150))
            if plot_proc[0][-4:] == 'corr':
                plt.ylim((-40,40))
            else:
                pass
                plt.ylim((-90,-10))
            #plt.plot(exp_results[v_stim]['F']['F'],exp_results[v_stim]['F']['PolyItself']) #This seems to be doing something weird even in the left channel, should check
            plt.subplot(2,2,3)
            plt.plot(exp_results[v_stim]['Osc']['Basis'],exp_results[v_stim][plot_proc[1]]['State'][:,side_idx[do_side]],linewidth=3)
            
            if plot_proc[0][-4:] == 'corr':
                plt.ylim((-20,20))
            else:
                pass
                plt.ylim((-70,-30))
            
            
            plt.subplot(2,2,4)
            plt.scatter(v_stim,exp_results[v_stim]['GCratio'][side_idx[do_side]])
            plt.ylim((0.8,2.2))
            
        plt.legend(do_stimvs)
        plt.suptitle(gel + str(plot_proc) + do_side + '| Bands: ' + band_approach + ' Calc: ' + do_calc)
        