# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import struct
import os

def ReadFile(filepath,size):
    binfile = open(filepath, 'rb')
    s = os.path.getsize(filepath)
    if size > s:
      size = s
    data = struct.unpack('f'*size,binfile.read(4*size))
    binfile.close()
    return data

def countpeak(histd,n_group):
  count = 0
  for i in range(0,n_group-1):
    if histd[i]>histd[i+1] and histd[i]>20:
      if i==0:
        count=count+1
      else:
        if histd[i]>histd[i-1]:
          count=count+1
  return count


route = '/home/guyu/tongji/SOMNN+SVM/data/n10mf10m/'
filenamer = ['QAM16','QAM64','QAM256','BPSK','QPSK','GMSK','OFDM']
filetype = '.bin'
plotflag = 1
filterflag = 1
category = 7
n_sample = 3000*2
traindataset = 100
simdataset = 100
ndataset = traindataset + simdataset
n_group=100
group = np.linspace(0,1,n_group+1)
groupplot = np.linspace(-1,1-1./n_group,2*n_group)
data = [[]]*category
train_n = [[]]*category
sim_n = [[]]*category
N=5 #change N for better performance
weights=np.hanning(N)
weightsphase=np.hanning(5)
for j in range(0,category):
  data[j] = (ReadFile(route+filenamer[j]+filetype,n_sample*ndataset))
  #train
  d = np.mat([[data[j][i]+1j*data[j][i+1]] for i in range(0, n_sample*traindataset-1, 2)])
  distance = np.abs(d)
  distance = distance/distance.max()
  distance = distance.reshape((traindataset,n_sample/2))
  
  angle = np.angle(d)
  angle = (angle+np.pi)/(2*np.pi)
  angle = angle.reshape((traindataset,n_sample/2))
  
  histd, bin_edges = np.histogram(distance[0],bins=group)
  if filterflag:
    histd=np.convolve(weights/weights.sum(),histd,'same')
  countd = countpeak(histd,n_group)
  hista, bin_edges = np.histogram(angle[0],bins=group)
  if filterflag:
    hista=np.convolve(weightsphase/weightsphase.sum(),hista,'same')
  counta = countpeak(hista,n_group)
  
  histd = histd/float(histd.max())
  hista = hista/float(hista.max())
  temp = np.hstack((countd,100*np.var(hista)))#remove counta, 100*np.var(histd),np.mean(distance[0])
  for i in range(1,traindataset):
    histd, bin_edges = np.histogram(distance[i],bins=group)
    if filterflag:
      histd=np.convolve(weights/weights.sum(),histd,'same')
    countd = countpeak(histd,n_group)
    hista, bin_edges = np.histogram(angle[i],bins=group)
    if filterflag:
      hista=np.convolve(weightsphase/weightsphase.sum(),hista,'same')
    counta = countpeak(hista,n_group)
    
    histd = histd/float(histd.max())
    hista = hista/float(hista.max())
    temp = np.vstack((temp,np.hstack((countd,100*np.var(hista)))))#remove counta, 100*np.var(histd),np.mean(distance[i])
  train_n[j] = temp
    
  #sim
  d = np.mat([[data[j][i]+1j*data[j][i+1]] for i in range(n_sample*traindataset, n_sample*ndataset-1, 2)])
  
  distance = np.abs(d)
  distance = distance/distance.max()
  distance = distance.reshape((simdataset,n_sample/2))
  
  angle = np.angle(d)
  angle = (angle+np.pi)/(2*np.pi)
  angle = angle.reshape((simdataset,n_sample/2))
  
  histd, bin_edges = np.histogram(distance[0],bins=group)
  if filterflag:
    histd=np.convolve(weights/weights.sum(),histd,'same')
  countd = countpeak(histd,n_group)
  hista, bin_edges = np.histogram(angle[0],bins=group)
  if filterflag:
    hista=np.convolve(weightsphase/weightsphase.sum(),hista,'same')
  counta = countpeak(hista,n_group)
  
  histd = histd/float(histd.max())
  hista = hista/float(hista.max())
  temp = np.hstack((countd,100*np.var(hista)))#remove counta, 100*np.var(histd),np.mean(distance[0])
  for i in range(1,simdataset):
    histd, bin_edges = np.histogram(distance[i],bins=group)
    if filterflag:
      histd=np.convolve(weights/weights.sum(),histd,'same')
    countd = countpeak(histd,n_group)
    hista, bin_edges = np.histogram(angle[i],bins=group)
    if filterflag:
      hista=np.convolve(weightsphase/weightsphase.sum(),hista,'same')
    counta = countpeak(hista,n_group)
  
    histd = histd/float(histd.max())
    hista = hista/float(hista.max())
    temp = np.vstack((temp,np.hstack((countd,100*np.var(hista)))))#remove counta, 100*np.var(histd),np.mean(distance[i])
  sim_n[j] = temp

#SVM
xtrain = train_n[0]
ytrain = np.zeros((traindataset,1))
for j in range(1,category):
  xtrain = np.vstack((xtrain,train_n[j]))
  ytrain = np.vstack((ytrain,j*np.ones((traindataset,1))))
ytrain = ytrain.ravel()
classifier = SVC(kernel='rbf',gamma='scale', C=1,decision_function_shape='ovo').fit(xtrain,ytrain)

xtest = sim_n[0]
ytest = np.zeros((1,simdataset))
for j in range(1,category):
  xtest = np.vstack((xtest,sim_n[j]))
  ytest = np.hstack((ytest,j*np.ones((1,simdataset))))
ytest = ytest.ravel()
y = classifier.predict(xtest)
print y
print ytest
print(np.sum(y==ytest)/(float(category)*simdataset))

'''
for root,dirs,files in os.walk('/home/guyu/tongji/SOMNN+SVM/data'):
     for dir in dirs:
             print dir
             print(os.path.join(root,dir))
'''
