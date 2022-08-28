#%%
"""
Created on Mon Dec 16 18:07:46 2019

@author: virati
Splitting our simulation of the dLFP into separate script
"""

from dbspace.signal.dLFP.diff_amp_model import diff_amp
from dbspace.signal.dLFP.sig_amp_model import sig_amp
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 10)

#%% [markdown]

# # MC Simulator
# This notebook simulates dLFP at different impedance mismatches.

#%%
# Parameter set
Z1, Z3 = 1200, 1300
stimulation_frequency = 130
amp_model = "tanh"

#%%[markdown]

# ## Let's simulate!
# At this point we're going to simulate the parameters set above


#%%
diff_run = diff_amp(
    Ad=200, wform="moresine4", clock=True, stim_v=4, stim_freq=stimulation_frequency
)
amp_run = sig_amp(
    diff_run, family=amp_model, noise=1e-6, sig_amp_gain=10, pre_amp_gain=5
)

amp_run.simulate(Z1, Z3, use_windowing="blackmanharris")
amp_run.plot_time_dom()
amp_run.plot_freq_dom()
amp_run.plot_tf_dom()

plt.show()
