#%%
"""
Created on Mon Dec 16 18:07:46 2019

@author: virati
Splitting our simulation of the dLFP into separate script
"""

import dbspace.signal.dLFP.diff_amp_model as diff_model
import dbspace.signal.dLFP.sig_amp_model as sig_amp
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 10)

#%% [markdown]

# # MC Simulator
# This notebook simulates dLFP at different impedance mismatches.

#%%
# Parameter set
Z1, Z3 = 1300, 1300
stimulation_frequency = 130
amp_model = "tanh"

#%%[markdown]

# ## Let's simulate!
# At this point we're going to simulate the parameters set above


#%%
diff_run = diff_model.sim_diff(
    Ad=10,
    wform="moresine4",
    clock=True,
    stim_v=4,
    stim_freq=stimulation_frequency,
    full_Fs=10000,
)
amp_run = sig_amp.sim_amp(
    diff_run, family=amp_model, noise=1e-6, sig_amp_gain=10, pre_amp_gain=0.1
)

amp_run.simulate(Z1, Z3)
amp_run.plot_simulation()
amp_run.plot_time_dom()
amp_run.plot_freq_dom()
amp_run.plot_tf_dom()

plt.show()

# %%
