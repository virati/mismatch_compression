#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 18:09:48 2020

@author: virati
Look at stimulation waveform from BlackRock 30kHz sampling
"""

import matplotlib.pyplot as plt
import sys
sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/NPMK/')

import pdb
from numpy               import arange
from brpylib             import NsxFile, brpylib_ver
import scipy.signal as sig
import numpy as np
import scipy
import scipy.stats as stats
import scipy.io

# Version control
brpylib_ver_req = "1.3.1"
if brpylib_ver.split('.') < brpylib_ver_req.split('.'):
    raise Exception("requires brpylib " + brpylib_ver_req + " or higher, please use latest version")

plt.rcParams['image.cmap'] = 'jet'

datafile = '/home/virati/MDD_Data/OR/DBS906/20150727-174713/20150727-174713-001.ns6'
#datafile = '/home/virati/MDD_Data/OR/DBS906/20150727-174713/20150727-174713-001.ns6'
home = '/home/virati/MDD_Data/OR/'
pt = 'DBS906'


#datafile = home + pt + file
#datafile = '/home/virati/MDD_Data/OR/DBS906//20150727-181004/20150727-181004-001.ns2'

elec_ids     = 'all'  # 'all' is default for all (1-indexed)
start_time_s = 500                       # 0 is default for all
data_time_s  = 500                     # 'all' is default for all
downsample   = 1                     # 1 is default
#plot_chan    = 5                       # 1-indexed

# Open file and extract headers
nsx_file = NsxFile(datafile)

# Extract data - note: data will be returned based on *SORTED* elec_ids, see cont_data['elec_ids']
cont_data = nsx_file.getdata(elec_ids, start_time_s, data_time_s, downsample)

# Close the nsx file now that all data is out
nsx_file.close()
#%%

Fs = cont_data['samp_per_s']
tser = cont_data['data']
plt.figure()
plt.plot(tser[30,0:10000])