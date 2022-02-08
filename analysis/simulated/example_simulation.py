#%%
"""
Created on Mon Dec 16 18:07:46 2019

@author: virati
Splitting our simulation of the dLFP into separate script
"""

import dbspace.signal.dLFP.diff_model as diff_model
from dbspace.signal.dLFP.diff_model import sim_diff, sim_amp
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 10)

#%%
diff_run = sim_diff(Ad=200, wform="moresine4", clock=True, stim_v=4, stim_freq=130)
amp_run = sim_amp(diff_run, family="tanh", noise=1e-6, sig_amp_gain=10, pre_amp_gain=10)
Z1, Z3 = 1200, 1300

amp_run.simulate(Z1, Z3, use_windowing="blackmanharris")
amp_run.plot_time_dom()
amp_run.plot_freq_dom()
amp_run.plot_tf_dom()

plt.show()
