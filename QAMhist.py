# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 12:36:18 2018
@author: Heisenberg
https://blog.csdn.net/qq_42633819/article/details/82942561
"""
 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import struct
import os
import sys

def ReadFile(filepath,size):
  binfile = open(filepath, 'rb')
  s = os.path.getsize(filepath)
  if size > s:
    size = s
  data = struct.unpack('f'*size,binfile.read(4*size))
  binfile.close()
  return data

plotflag=0

route = '/home/guyu/tongji/SOMNN+SVM/data/n10mf10m/'
filenamer = ['QAM16','QAM64','QAM256','BPSK','QPSK','GMSK','OFDM','OFDMQPSK','OFDMQPSK8PSK']
filetype = '.bin'
category = 7
size = 3000*2
ndataset = 100
data = [[]]*category
n_group=100
group = np.linspace(0,1,n_group)
countd=np.zeros((category,ndataset))
counta=np.zeros((category,ndataset))
vara=np.zeros((category,ndataset))
vard=np.zeros((category,ndataset))
varf=np.zeros((category,ndataset))
varfhanning=np.zeros((category,ndataset))
N=5 #change N for better performance
weights=np.hanning(N)
weightsphase=np.hanning(25)
weightsfreq=np.hanning(25)
for ii in range(0,ndataset):
  for j in range(0,category):
    data[j] = (ReadFile(route+filenamer[j]+filetype,size*ndataset))
    d = np.mat([[data[j][ii*size+i]+1j*data[j][ii*size+i+1]] for i in range(0, size-1, 2)])
    # frequency
    f=np.arange(size/2)
    freq=np.fft.fftshift(np.fft.fft(d))
    freq = np.reshape(freq,(1,size/2))
    freq = freq[0]
    absfreq = np.abs(freq)
    absfreqhanning=np.convolve(weightsfreq/weightsfreq.sum(),absfreq,'same')
    varf[j][ii]=np.var(absfreq)
    varfhanning[j][ii]=np.var(absfreqhanning)
    
    # distance
    distance=np.abs(d)
    distance = distance/distance.max()
    if plotflag==2 and ii<1:
      plt.figure()
      plt.subplot(1, 2, 1)
      plt.scatter(d.real, d.imag, c="red",s=30, edgecolor='k')
      plt.subplot(1, 2, 2)
      plt.scatter(d-d, distance, c="red",s=30, edgecolor='k')  
      plt.suptitle(filenamer[j])   
      plt.show() 
    
    histd, bin_edges = np.histogram(distance,bins=n_group)
    vard[j][ii]=np.var(histd)
    histd=np.convolve(weights/weights.sum(),histd,'same')
    for i in range(0,n_group-1):
      if histd[i]>histd[i+1] and histd[i]>20:
        if i==0:
          countd[j][ii]=countd[j][ii]+1
        else:
          if histd[i]>histd[i-1]:
            countd[j][ii]=countd[j][ii]+1
    # angle
    angle=np.angle(d)
    angle = (angle+np.pi)/(2*np.pi)
    hista, bin_edges = np.histogram(angle,bins=n_group)
    vara[j][ii]=np.var(hista)
    hista=np.convolve(weightsphase/weightsphase.sum(),hista,'same')
    for i in range(0,n_group-1):
      if hista[i]>hista[i+1] and hista[i]>20:
        if i==0:
          counta[j][ii]=counta[j][ii]+1
        else:
          if hista[i]>hista[i-1]:
            counta[j][ii]=counta[j][ii]+1

    # plot
    if plotflag==1 and ii<1: 
      hang = 3
      lie = 2    
      plt.figure(figsize=(6,4))
      plt.subplot(hang,lie,1)
      plt.hist(angle, group, histtype='bar')
      plt.title('phase',fontsize = 15,fontweight = 'normal')
      plt.subplot(hang,lie,2)
      plt.bar(group,hista,align='center',width=1.0/n_group)
      plt.title('phase hanning',fontsize = 15,fontweight = 'normal')
      plt.tight_layout()
            
      plt.subplot(hang,lie,3)
      plt.hist(distance, group, histtype='bar')
      plt.title('amplitude',fontsize = 15,fontweight = 'normal')
      plt.subplot(hang,lie,4)
      plt.bar(group,histd,align='center',width=1.0/n_group)
      plt.title('amplitude hanning',fontsize = 15,fontweight = 'normal')
   
      plt.subplot(hang,lie,5)
      plt.plot(f,absfreq)
      plt.title('frequency',fontsize = 15,fontweight = 'normal')
      plt.subplot(hang,lie,6)
      plt.plot(f,absfreqhanning)
      plt.title('frequency hanning',fontsize = 15,fontweight = 'normal')
      
      plt.tight_layout()
      
# normalization
countd = countd/float(np.max(countd))*100
counta = counta/float(np.max(counta))*100
vard = vard/float(np.max(vard))*100
vara = vara/float(np.max(vara))*100
varf = varf/float(np.max(varf))*100
varfhanning = varfhanning/float(np.max(varfhanning))*100


np.set_printoptions(precision=2)       
print 'distance'
print np.min(countd,1)
print np.max(countd,1)

print 'angle'
print np.min(counta,1)
print np.max(counta,1)

print 'vard'
print np.min(vard,1)
print np.max(vard,1)

print 'vara'
print np.min(vara,1)
print np.max(vara,1)

print 'varf'
print np.min(varf,1)
print np.max(varf,1)

print 'varfhanning'
print np.min(varfhanning,1)
print np.max(varfhanning,1)

if plotflag==1:
  plt.show()





