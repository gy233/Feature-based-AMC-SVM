# -*- coding: utf-8 -*-
'''
AUTHOR: GUYU
AT HUANGDU SCIENCE AND TECHNOLOGY COLLEGE
2020
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import struct
import os,xlrd,xlwt
from xlutils.copy import copy

def ReadFile(filepath,size):
  print(filepath)
  binfile = open(filepath, 'rb')
  s = os.path.getsize(filepath)
  if size > s:
    size = s
  data = struct.unpack('f'*size,binfile.read(4*size))
  binfile.close()
  return data


#route = '/home/guyu/tongji/SOMNN+SVM/simdata_agc/n3160mf200m/'
route = '/home/guyu/tongji/SOMNN+SVM/data/real_agc/'
filenamer = ['BPSK','QPSK','QAM16','QAM64','QAM256','GMSK','OFDM']
filetype = '.bin'
plotflag = 0
category = 7
n_sample = 3000*2
traindataset = 100
simdataset = 100
deletedataset = 15
ndataset = traindataset + simdataset + deletedataset
n_group=100
group = np.linspace(0,1,n_group+1)
groupplot = np.linspace(-1,1-1./n_group,2*n_group)
train_n = [[]]*category
sim_n = [[]]*category
N=5 #change N for better performance
weights=np.hanning(N)
for j in range(0,category):
  data = (ReadFile(route+filenamer[j]+filetype,n_sample*ndataset))
  #train
  d = np.mat([[data[i]+1j*data[i+1]] for i in range(0, n_sample*(traindataset+deletedataset)-1, 2)])
  d = d[n_sample*deletedataset/2:]
  d = d.reshape((traindataset,n_sample/2))
  
  distance = np.abs(d)
  distance = distance/distance.max(1).astype(np.float)
  
  angle = np.angle(d)
  angle = (angle+np.pi)/(2*np.pi)
  
  freq = np.abs(np.fft.fft(d))
  freqmax = freq.max(1)
  for i in range(0,len(freqmax)):
    freq[i] = freq[i]/float(freqmax[i])
  
  histd, bin_edges = np.histogram(distance[0],bins=group)
  hista, bin_edges = np.histogram(angle[0],bins=group)
  temp = np.hstack((histd,hista))
  #temp = np.hstack((histd,hista,freq[0]))
  for i in range(1,traindataset):
    histd, bin_edges = np.histogram(distance[i],bins=group)
    hista, bin_edges = np.histogram(angle[i],bins=group)
    temp = np.vstack((temp,np.hstack((histd,hista))))
    #temp = np.vstack((temp,np.hstack((histd,hista,freq[i]))))
  train_n[j] = temp
    
  #sim
  d = np.mat([[data[i]+1j*data[i+1]] for i in range(n_sample*(traindataset+deletedataset), n_sample*ndataset-1, 2)])
  d = d.reshape((simdataset,n_sample/2))
  
  distance = np.abs(d)
  distance = distance/distance.max(1).astype(np.float)
  
  angle = np.angle(d)
  angle = (angle+np.pi)/(2*np.pi)
  
  freq = np.abs(np.fft.fft(d))
  freqmax = freq.max(1)
  for i in range(0,len(freqmax)):
    freq[i] = freq[i]/float(freqmax[i])
  
  histd, bin_edges = np.histogram(distance[0],bins=group)
  hista, bin_edges = np.histogram(angle[0],bins=group)
  temp = np.hstack((histd,hista))
  #temp = np.hstack((histd,hista,freq[0]))
  for i in range(1,simdataset):
    histd, bin_edges = np.histogram(distance[i],bins=group)
    hista, bin_edges = np.histogram(angle[i],bins=group)
    temp = np.vstack((temp,np.hstack((histd,hista))))
    #temp = np.vstack((temp,np.hstack((histd,hista,freq[i]))))
  sim_n[j] = temp


if plotflag==1:      
  for j in range(0,category):
    plt.figure(figsize=(10, 8))
    for i in range(1,5):
      plt.subplot(2,2,i)
      plt.bar(groupplot,np.array(train_n[j][i]),width=1/float(n_group),align='edge')
    plt.suptitle(filepathr[j]) 
plt.show()

#neural_network
xtrain = train_n[0]
ytrain = np.zeros((traindataset,1))
for j in range(1,category):
  xtrain = np.vstack((xtrain,train_n[j]))
  ytrain = np.vstack((ytrain,j*np.ones((traindataset,1))))
ytrain = ytrain.ravel()
clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=50, random_state=1, activation='relu')
clf.fit(xtrain, ytrain)
xtest = sim_n[0]
ytest = np.zeros((1,simdataset))
for j in range(1,category):
  xtest = np.vstack((xtest,sim_n[j]))
  ytest = np.hstack((ytest,j*np.ones((1,simdataset))))
ytest = ytest.ravel()

y=clf.predict(xtest)
for j in range(0,category):
  for i in range(0,category):
    print('y='+str(i)+',ytest='+str(j)+': ',np.sum(np.logical_and(y==i, ytest==j))/(float(simdataset)))
print(np.sum(y==ytest)/(float(category)*simdataset))
if plotflag==1:
  plt.show()


if 'simdata_agc' in route:
  xls_path = '/home/guyu/tongji/SOMNN+SVM/result/histNN_result_agc.xls'
elif 'simdata_withoutagc' in route:
  xls_path = '/home/guyu/tongji/SOMNN+SVM/result/histNN_result_withoutagc.xls'
else:
 xls_path = '/home/guyu/tongji/SOMNN+SVM/result/histNN_result.xls'
if not os.path.exists(xls_path):
  book = xlwt.Workbook(encoding='utf-8')
  book.add_sheet('Sheet1')
  book.save(xls_path)
  print(xls_path+' is created')
wb = xlrd.open_workbook(xls_path) 
sheetname = route.split('/')[6]
if not(sheetname in wb.sheet_names()):
  workbook = copy(wb)
  sheet = workbook.add_sheet(sheetname, cell_overwrite_ok=True)
  for j in range(0,category):
    for i in range(0,category):
      sheet.write(j+1,i+1,np.sum(np.logical_and(y==i, ytest==j))/(float(simdataset)))
  for j in range(0,category):
      sheet.write(j+1,0,filenamer[j])
      sheet.write(0,j+1,filenamer[j])
  sheet.write(category+2,0,'total')
  sheet.write(category+2,1,np.sum(y==ytest)/(float(category)*simdataset))
  workbook.save(xls_path)
  print(sheetname+' has been added successfully')
else:
  print(sheetname+' exists in the file')
