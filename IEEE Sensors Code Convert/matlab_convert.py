#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 23:17:53 2020

@author: Kayvon
"""
from PyEMD import EEMD
import pandas as pd
from scipy import signal
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
#import obspy

fs = 100

### Read data
sig = pd.read_csv('/Users/Kayvon/Downloads/2020-IEEE-Sensors-Journal-master/joseHB.csv')
sigj = sig.iloc[4000:10000]
sigj = np.ravel(a=sigj)

### FIR filter to remove high frequency noises
sigf = signal.savgol_filter(x=sigj,window_length=5,polyorder=3)

### Plot filtering result
# plt.plot(sigj)
# plt.plot(sigf)
# plt.show()

### EMD component extraction
s = sigf
eemd = EEMD(trials=200,noise_width=0.1)
eIMFs = eemd(s,max_imf=1000)
# plt.plot(eIMFs)
# plt.show()
nIMFs = eIMFs.shape[0]

## Calculate the dominant frequencies of different imfs
analytic_signal = hilbert(eIMFs)
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = (np.diff(instantaneous_phase)/((2.0*np.pi) * fs))

#instantaneous_frequency = obspy.signal.cpxtrace.instantaneous_frequency(eIMFs,fs=fs)

ff = []

for x in range(11):
    mean = np.mean(instantaneous_frequency[x,:])
    ff.append(mean)

# plt.plot(ff)
# plt.show()
###    
Hrlow = 0.75
Hrhigh = 2
Rrlow = 0.05
Rrhigh = 0.75

### NOT CURRENTLY WORKING
for y in range(len(ff)):
    HRIndex = np.where(ff[y]>Hrlow and ff[y]<Hrhigh) 
    RRIndex = np.where(ff[y]>Rrlow and ff[y]<Rrhigh)

### HR pca
ftlength = 10000
df = fs/ftlength


pca = PCA()
pca.fit(eIMFs[4,:].reshape(-1,1))
#pca_score = pca.score(eIMFs[:,4].reshape(-1,1))
HR_pca_score = pca.fit_transform(eIMFs[4].reshape(-1,1))

hrspectrum = abs(np.fft.fft(HR_pca_score[:,0],n=ftlength))
hrspectrumf = signal.savgol_filter(x=hrspectrum,window_length=5,polyorder=3)

### RR pca
# for x in range(5,11,1):
#     pca.fit(eIMFs[x,:].reshape(-1,1))
#     RR_pca_score = pca.fit_transform(eIMFs[x].reshape(-1,1))

pca.fit(eIMFs[5,:].reshape(-1,1))
RR_pca_score = pca.fit_transform(eIMFs[5].reshape(-1,1))

rrspectrum = abs(np.fft.fft(RR_pca_score[:,0],n=ftlength))
rrspectrumf = signal.savgol_filter(x=rrspectrum,window_length=5,polyorder=3)

### Calculate HR and RR
hhlow = Hrlow/df
hhhigh = Hrhigh/df
rrlow = Rrlow/df
rrhigh = Rrhigh/df

HR = np.argmax(hrspectrumf[int(hhlow):int(hhhigh)], axis=0)
HR = HR + hhlow - 2
HR = HR*df*60
RR = np.argmax(hrspectrumf[int(rrlow):int(rrhigh)], axis=0)
RR = RR + rrlow - 2
RR = RR*df*60
print('HR = '+str(HR))
print('RR = '+str(RR))



ddf1 = np.linspace(int(0),int(100),int(10000))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.plot(ddf1,hrspectrum[:],color='blue',markersize=0.01)
ax.plot(ddf1,hrspectrumf[:],color='red',markersize=0.01)
ax.plot(ddf1,rrspectrum[:],color='yellow',markersize=0.01)
ax.plot(ddf1,rrspectrumf[:],color='purple',markersize=0.01)
ax.set_xlim(0,2)
ax.set_facecolor('xkcd:light grey')


