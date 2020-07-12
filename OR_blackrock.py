# -*- coding: utf-8 -*-
"""
Example of how to extract and plot continuous data saved in Blackrock nsX data files

@author: Mitch Frankel - Blackrock Microsystems

Version History:
v1.0.0 - 07/05/2016 - initial release - requires brpylib v1.0.0 or higher
v1.1.0 - 07/12/2016 - addition of version checking for brpylib starting with v1.2.0
                      minor code cleanup for readability
v1.1.1 - 07/22/2016 - now uses 'samp_per_sec' as returned by NsxFile.getdata()
                      minor modifications to use close() functionality of NsxFile class
                      
Modified by vineet tiruvadi, for Helen Mayberg (Emory) DBS Study - OR DATA
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

# Inits
#datafile = 'C:/Users/mfrankel/Dropbox/BlackrockDB/software/sampledata/The Most Perfect Data in the WWWorld/' \'sampleData.ns6'
datafile = '/run/media/virati/Stokes/MDD_Data/OR/DBS905/20150831-164204-002.ns2'
#datafile = '/run/media/virati/Stokes/MDD_Data/OR/DBS906/20150727-174713/20150727-174713-001.ns2' 
#datafile = '/mnt/auto/MDD_Data/Active/OR/DBS906/20150727-174713/20150727-174713-001.ns6'
#datafile = '/home/virati/MDD_Data/OR/DBS906/20150727-174713/20150727-174713-001.ns6'
home = '/home/virati/MDD_Data/OR/'
pt = 'DBS906'
file = '/20150727-130014/20150727-130014-001.ns2'

#datafile = home + pt + file
#datafile = '/home/virati/MDD_Data/OR/DBS906//20150727-181004/20150727-181004-001.ns2'

elec_ids     = 'all'  # 'all' is default for all (1-indexed)
start_time_s = 1115                       # 0 is default for all
data_time_s  = 300                     # 'all' is default for all
downsample   = 1                       # 1 is default
#plot_chan    = 5                       # 1-indexed

# Open file and extract headers
nsx_file = NsxFile(datafile)

# Extract data - note: data will be returned based on *SORTED* elec_ids, see cont_data['elec_ids']
cont_data = nsx_file.getdata(elec_ids, start_time_s, data_time_s, downsample)

# Close the nsx file now that all data is out
nsx_file.close()
#%%

def plot_PSDs(znorm=False):
    for ii,plot_chan in enumerate(elec_ids):
        # Plot the data channel
        ch_idx  = cont_data['elec_ids'].index(plot_chan)
        hdr_idx = cont_data['ExtendedHeaderIndices'][ch_idx]
        t       = cont_data['start_time_s'] + arange(cont_data['data'].shape[1]) / cont_data['samp_per_s']
        ch_name = nsx_file.extended_headers[hdr_idx]['ElectrodeLabel']
        
        if znorm:
            print('Zscoring')
            ts_in = stats.zscore(cont_data['data'][ch_idx])
        else:
            ts_in = cont_data['data'][ch_idx]
        
        plt.subplot(311)
        plt.plot(t, ts_in,label=ch_name)
        #plt.axis([t[0], t[-1], min(ts_in), max(cont_data['data'][ch_idx])])
        plt.locator_params(axis='y', nbins=20)
        plt.xlabel('Time (s)')
        plt.ylabel("Output (" + nsx_file.extended_headers[hdr_idx]['Units'] + ")")
        plt.title(nsx_file.extended_headers[hdr_idx]['ElectrodeLabel'])
        
        F,T,SG = sig.spectrogram(ts_in,nperseg=256,noverlap=250,window=sig.get_window('blackmanharris',256),fs=500)
        plt.subplot(312)
        plt.plot(F,np.median(10*np.log10(SG),axis=1),label=ch_name)
        plt.show()
    
    plt.legend()
    
    plt.subplot(313)
    ch_1 = cont_data['elec_ids'].index(elec_ids[2])
    ch_3 = cont_data['elec_ids'].index(elec_ids[3])
    
    print(ch_1)
    ts_1 = cont_data['data'][ch_1]
    ts_3 = cont_data['data'][ch_3]
    
#ch_names = plot_PSDs(znorm=False)
#%%

def plot_TS(plotch = 30):
    plt.figure()
    fs = cont_data['samp_per_s']
    ds_f = 10

    #tvect = cont_data['start_time_s'] + arange(cont_data['data'].shape[1]) / (cont_data['samp_per_s'] / ds_f)
    sigin = cont_data['data'][plotch,:] - cont_data['data'][0,:]
    ts_sig = sig.decimate(sigin,ds_f)
    tvect = np.linspace(cont_data['start_time_s'],cont_data['start_time_s'] + ts_sig.shape[0]/(fs/ds_f),ts_sig.shape[0])
    plt.title(plotch)
    
    #pdb.set_trace()
    plt.plot(tvect,ts_sig)
    #plt.ylim((0,100))

plot_TS()


#%%
def plot_SGs(plotch = 30):
    #find which channel is associated with LFPs
    LFPchanns = []
    #for aa,chname in cont_data['elec_ids']:
    #    if chname[0:3] == 'LFP':
    #        LFPchanns.append(aa)
    
    #np.min(LFPchanns)
    plt.figure()
    fs = cont_data['samp_per_s']
    ds_f = 1
    sigin = cont_data['data'][plotch,:] - cont_data['data'][0,:]
    ts_SG = sig.decimate(sigin,ds_f)
    F,T,SG = sig.spectrogram(ts_SG,nperseg=1024,noverlap=512,window=sig.get_window('blackmanharris',1024),fs=fs/ds_f)
    plt.title(plotch)
    #plt.pcolormesh(T,F,np.log10(SG))
    #plt.plot(F,np.median(10*np.log10(SG),axis=1))
    plt.pcolormesh(T+start_time_s,F,np.log10(SG),rasterized=True)
    plt.colorbar()
    #plt.ylim((0,100))
    #np.save('/tmp/blackrock_sample.npy',cont_data['data'])
    print(cont_data['data'].shape)
plot_SGs(plotch = 30) #30 has a chirp, so probably LFP
    

