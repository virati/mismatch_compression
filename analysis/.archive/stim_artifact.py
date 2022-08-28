#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 13:55:26 2017

@author: virati
Stimulation artifact plotting and outlining
DISSERTATION FINAL
"""
# import Sim_Sig as SiSi
import numpy as np

# import prettyplotlib as ppl
import matplotlib.pyplot as plt
import matplotlib
import scipy.signal as sig
import scipy.io as scio
import pandas as pd
import scipy

from collections import defaultdict

# import dill

from mpl_toolkits.mplot3d import Axes3D


def gcd(a, b):
    while b > 0:
        a, b = b, a % b
    return a


def lcm(a, b):
    return a * b / gcd(a, b)


def compute_GC_metric(F, Pxx):
    # compute the ratio of 64Hz to 130Hz to get "level of compression"
    F_harm2 = np.where(np.logical_and(F > 62, F < 64))
    F_harm1 = np.where(np.logical_and(F > 30, F < 34))

    # pow_ratio = 10*(np.log10(np.median(Pxx[F_harm2])) - np.log10(np.median(Pxx[F_harm1])))
    pow_ratio = np.log10(np.median(Pxx[F_harm2])) / np.log10(np.median(Pxx[F_harm1]))
    # pow_ratio = np.median(Pxx[F_harm2])/np.median(Pxx[F_harm1])
    return {"Label": "Ratio of 2nd Harmonic and 1st Harmonic", "Ratio": pow_ratio}


font = {"family": "normal", "weight": "bold", "size": 22}

matplotlib.rc("font", **font)

plt.rcParams["image.cmap"] = "jet"

# If we want quick plotting from scratch, we use this; this may be more appropriate for just talking about stimulation artifacts
# Fs = 1000
# t = np.linspace(-10,10,Fs*20)
# x_sin = np.sin(2 * np.pi * 130 * t)
# x = np.array((sig.square(2 * np.pi * 130 * t, duty=10*(90e-6)/(1/130))),dtype=float)
stim = "ipg"
Fs = 4220
brFs = 422
# br_x = sig.resample_poly(x,211000/Fs,211000/422)
# br_t = np.linspace(-10,10,br_x.shape[0])

big_t = np.linspace(-10, 10, Fs * 20)
if stim == "sine":
    big_x = 10 * np.sin(2 * np.pi * 130 * big_t)
elif stim == "square":
    big_x = 10 * (
        np.array(
            sig.square(2 * np.pi * 130 * big_t, duty=10 * (90e-6) / (1 / 130)),
            dtype=float,
        )
        / 2
        + 1 / 2
    )
elif stim == "real":
    # Comes in with 1000 Hz sampling of a 130Hz
    # 100 -> 7790 is peak to peak
    # so 7690 corresponds to 0.00769 (or 1/130) seconds
    # needs to go to 32.46 samples
    # sampling rate currently is 100,000Hz

    # this needs to become ~32.46 samples peak-peak at 4220 Hz
    stim_wf_matrix = scio.loadmat("/tmp/waveforms.mat")["y"][3]
    # resamp = sig.resample(stim_wf_matrix[100:7790],65)
    dsamp = sig.decimate(stim_wf_matrix, 16)
    dsamp = sig.decimate(dsamp, 14)
    # repeat this up to what we want
    # OR JUST DO THIS IN THE SALINE AND EXTRACT LITERALLY WHAT THE STIM IS ALONE***********
    big_x = np.tile(dsamp, 2638)[0 : big_t.shape[0]]
elif stim == "ipg":
    ipg_infile = "/home/vscode/data/stim_waveform/ssipgwave_vreg_Ra1p1kOhm_1usdt.txt"
    inmatr = np.array(pd.read_csv(ipg_infile, sep=",", header=None))

    # concatenate this to massive
    concatstim = np.tile(inmatr[:, 1], 223)
    ccstim_tvect = np.linspace(0, concatstim.shape[0] / 1e6, concatstim.shape[0]) - 10

    # now downsample this using interp
    artif = scipy.interpolate.interp1d(ccstim_tvect, concatstim)
    # save artif into a pickled function

    orig_big_x = artif(big_t)

# just save this file

# orig_big_x is the actual stimulation waveform. Maybe we can add stuff to it now???

br_t = big_t[0::10]

# gaussian blur the orig_big_x to
big_x = orig_big_x
# np.save('/tmp/ipg',orig_big_x)

# Plot big_x and its PSD
plt.subplot(211)
plt.plot(big_x)
plt.subplot(212)
F, Pxx = sig.welch(big_x, fs=Fs)
plt.plot(F, 10 * np.log10(Pxx), linewidth=5, label="Sampled")

#%%
br_x = big_x[0::10]


def do_sweep():

    g_sweep = np.linspace(0.01, 5, 500)
    # g_sweep = np.array([])

    measures = [None] * g_sweep.shape[0]

    for gg, g_fact in enumerate(g_sweep):
        big_x = np.tanh(g_fact * orig_big_x)
        # big_x = g_fact * orig_big_x

        # peak-to-peak of input voltage
        maxampl = np.max(np.abs(g_fact * big_x))
        maxoutampl = np.max(np.abs(big_x))

        # find the derivative of the tanh at the ptp val
        gc_level = 1 - np.tanh(maxampl) ** 2
        br_x = big_x[0::10]
        F, Pxx = sig.welch(br_x, fs=brFs)

        print("Input max: " + str(maxampl) + " gives GC level: " + str(gc_level))

        gc_measure = compute_GC_metric(F, Pxx)
        print(gc_measure)

        measures[gg] = {
            "MaxIn": maxampl,
            "MaxOut": maxoutampl,
            "gc_level": gc_level,
            "gc_metric": gc_measure,
        }

    #%%
    # plotting time!
    gc_fromg = np.array([rr["gc_metric"]["Ratio"] for rr in measures])
    inampl_fromg = np.array([rr["MaxIn"] for rr in measures])
    outampl_fromg = np.array([rr["MaxOut"] for rr in measures])
    trueg_fromg = np.array([rr["gc_level"] for rr in measures])

    plt.figure()
    ax1 = plt.subplot(211)
    ax1.plot(g_sweep, inampl_fromg.T, label="Input Amplitude")
    # ax1.plot(g_sweep,outampl_fromg.T,label='Max Output Amplitude')
    ax1.plot(g_sweep, inampl_fromg.T - outampl_fromg.T, label="Amplitude Deficit")
    adef_perc = (inampl_fromg.T - outampl_fromg.T) / inampl_fromg.T
    # ax1.plot(g_sweep,adef_perc,label='Amplitude Deficit %')

    ax1.set_ylabel("Voltage")
    plt.legend()
    ax4 = ax1.twinx()
    ax4.plot(g_sweep, trueg_fromg.T, label="GC Truth", color="green")
    ax4.set_ylabel("Gain Compression", color="g")
    ax4.tick_params("y", colors="g")
    plt.legend()

    ax2 = plt.subplot(212)
    ax2.plot(g_sweep, gc_fromg.T, label="GC Measure", color="red")
    ax2.set_ylabel("GC Measure (ratio of harmonics)", color="red")
    ax2.tick_params("y", colors="r")
    ax3 = ax2.twinx()
    ax3.plot(g_sweep, trueg_fromg.T, label="GC Truth", color="green")
    ax3.set_ylabel("Gain Compression", color="g")
    ax3.tick_params("y", colors="g")
    plt.axvline(x=g_sweep[96])
    plt.legend()
    #%%
    plt.figure()
    plt.subplot(211)
    plt.plot(g_sweep, np.hstack((np.diff(adef_perc), np.array([0]))))
    plt.title("First Derivative of amplitude deficit between input and output")
    # find max
    idx_inflection = np.argmax(np.diff(adef_perc))
    plt.subplot(212)
    plt.plot(
        g_sweep,
        np.hstack((np.diff(np.diff(adef_perc)), np.array([0, 0]))),
        label="Amplitude Deficit Deriv",
    )
    plt.title("Second Derivative of amplitude deficit between input and output")


#%%
np.save("/tmp/ipg.txt", br_x)
#%%
g_fact = 10
big_x = np.tanh(g_fact * orig_big_x)
br_x = np.copy(big_x[0::10])

plt.figure()
plt.subplot(311)
# first, we're plotting the fully sampled
plt.plot(big_t, big_x, color="b", alpha=0.5, linewidth=3)
# then we're plotting the 422 sampled ?
# plt.plot(br_t,br_x,alpha=0.5,linewidth=5,color='g')
plt.xlim((-10, -9.9))

markline, stemline, baseline = plt.stem(br_t[0:211], br_x[0:211])
# markline,stemline,baseline = plt.stem(br_t[0:1000],br_x[0:1000],alpha=0.2)
plt.setp(markline, "markerfacecolor", "g", "alpha", 0.9)
plt.setp(stemline, color="g", alpha=0.9)
plt.setp(stemline, "linewidth", 0.5)

plt.xlabel("Time")
plt.ylabel("Voltage")
plt.title("130Hz " + stim + " Stimulation and Sampled Stimulation")


def closeto(a, b, eps=0.01):
    # eps is percent
    return np.abs(a - b) < eps * a


plt.subplot(312)

F, Pxx = sig.welch(br_x, fs=brFs)
plt.plot(F, 10 * np.log10(Pxx), linewidth=5, label="Sampled")
# plt.ylim((0,0.004))
plt.legend()

ax = plt.subplot(313)
lns1 = ax.plot(F, Pxx, linewidth=5, label="Sampled", color="g")
# plt.ylim((0,0.004))

ax.legend()

# if you want it normalized somehow
# plt.plot(Fbig,PxxBig/(10*np.sum(PxxBig[Fbig < 200])),linewidth=3,label='Analog')
# if not

# find ratio of the 130Hz in sampled and analog; then just scale Analog up
# This is so they can be plotted on the same scale; should probably plot something on the right as well
Fbig, PxxBig = sig.welch(big_x, fs=Fs)
nPxxBig = (PxxBig / PxxBig[closeto(Fbig, 130, eps=0.02)]) * Pxx[closeto(F, 130)]

ax2 = ax.twinx()
lns2 = ax2.plot(Fbig, PxxBig, linewidth=3, label="Analog")

lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (dB)")
plt.title("PSD of the Signal")
plt.xlim((0, 1000))


#%%
# let's look solely at the  full waveform "presampled"
plt.figure()
plt.subplot(311)
# plt.plot(big_t,big_x)
plt.plot(inmatr[:, 1])
# plt.xlim((0,0.1))

plt.subplot(312)
F, Pxx = sig.welch(inmatr[:, 1], fs=1 / (1e-06))
plt.plot(F, Pxx)

plt.subplot(313)
plt.plot(F, np.log10(Pxx))

##%%
##Variable stim artifact leakage and the level of gain compression present

# Z = np.array([1000,1000])

# Z_1l = np.linspace(500,1500,1000)
# Z_3l = np.linspace(500,1500,1000)

# Z_1,Z_3 = np.meshgrid(Z_1l,Z_3l)

##%%
##Simple heatmap
##This assumes the current generating the LFP goes through a 1Ohm Resistor, just a normalization
# pre_gain = 250

# St = 6
# V_1 = 10e-6
# V_3 = 10e-6

##V_out = V_1*Z_1 - V_3*Z_3 + St*Z_1 - St*Z_3

# I_out = V_1/Z_1 - V_3/Z_3 + (St/Z_1 - St/Z_3)
##Assume the output impedance is low/one and we can do post-gains arbitrarily
# V_out = I_out

# def amp_TF(pre_gain,V_in):
# arb_out_gain = 1
# return arb_out_gain * np.tanh(pre_gain * V_in)

# plt.figure()
# max_Vout = np.abs(V_out)
# plt.subplot(221)
# plt.imshow(max_Vout,origin='lower',extent=[500,1500,500,1500])
# plt.colorbar()
# plt.xlabel('E_3 Impedance')
# plt.ylabel('E_1 Impedance')
# plt.title('Maximum Voltage Into Amplifier')

# plt.subplot(222)
# plt.imshow(amp_TF(2000,max_Vout),origin='lower',alpha=0.2)
##Calculate the noise cutoff
# noise_cutoff = 0.4
# invert_limit = np.arctanh(1 - noise_cutoff)
# plt.contour(amp_TF(2000,max_Vout),levels=[0,invert_limit,1],colors='r')
# plt.colorbar()
# plt.xlabel('E_3 Impedance')
# plt.ylabel('E_1 Impedance')
# plt.title('Maximum Voltage Into Amplifier')
# plt.title('Post Gain Tanh Saturation Level')


##%%

# def gaincompress(x,y):
# tot_gain = np.sqrt(np.abs(x*y))
# return np.tanh(1/(5*tot_gain) * (x-y))

# def intube(x,y):
# return x**2 + y

# zs = np.array([gaincompress(x,y)**2 for x,y in zip(np.ravel(Z_1l),np.ravel(Z_3l))])
# Z = zs.reshape(Z_1l.shape)

##%%
# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
##plt.contour(X,Y,Z,cmap=plt.cm.rainbow)
# ax.plot_surface(Z_1l,Z_2l,Z,cmap=plt.cm.viridis,alpha=0.5)
##put a horizontal plane at the point where the gain becomes ~80% of the original intended linear-gain region
##ax.plot_surface(Xm,Ym,0.002*np.ones_like(Z),alpha=0.2,color='red')
# ax.plot_wireframe(Xm,Ym,0.002*np.ones_like(Z),alpha=1,color='red',rstride=10,cstride=10)
##ax.plot_wireframe(Xm,Ym,np.reshape(Z[np.abs(Z-0.002)<0.0001],shape=Xm.shape))
##ax.contour(Xm,Ym,0.002*np.ones_like(Z))
# ax.set_zlabel('Amplifier Distortion')
# ax.set_xlabel('Electrode 1 Impedance')
# ax.set_ylabel('Electrode 3 Impedance')

##%%

##if we want to use the SimSig library, we do this
# if 0:
# kSig = SiSi.sim_sig()
# kSig.clear_sigs()
# kSig.add_stim_artifact(leak_factor=10,smod='sine',plot=True)


# #%%
# #Stim harmonics are at
# base_freq = 130
# ns = np.arange(1,10)

# shaping_harms = base_freq * ns

# #Pure sampled harmonic are at
# sampling_freq=422
# samp_ns = np.hstack((np.arange(-10,0),np.arange(1,10)))

# sampled_harms = np.zeros((shaping_harms.shape[0],samp_ns.shape[0]))
# harm_samped = np.zeros((shaping_harms.shape[0],samp_ns.shape[0]))
# for ss,sh in enumerate(shaping_harms):
#     sampled_harms[ss] = sampling_freq + samp_ns*sh
#     harm_samped[ss] = ss*np.ones_like(sampled_harms[ss])

# plt.subplot(211)
# plt.scatter(sampled_harms,1/(harm_samped+1))

# plt.subplot(212)
# #plt.plot()
