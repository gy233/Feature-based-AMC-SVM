# -*- coding: utf-8 -*-
'''
AUTHOR: GUYU
AT HUANGDU SCIENCE AND TECHNOLOGY COLLEGE
2020
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import struct
import os
import xlwt,xlrd
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

route = '/home/guyu/tongji/SOMNN+SVM/simdata_agc/n10mf200m/'
route = '/home/guyu/tongji/SOMNN+SVM/simdata_withoutagc/n10mf200m/'
#route = '/home/guyu/tongji/SOMNN+SVM/data/n316mf0m/'
#route = '/home/guyu/tongji/SOMNN+SVM/data/real_agc/'
#route = '/home/guyu/tongji/SOMNN+SVM/data/real_without_agc/'
filenamer = ['BPSK','QPSK','QAM16','QAM64','QAM256','GMSK','OFDM']
filetype = '.bin'
plotflag = 1
filterflag = 0
filteraflag = 1
category = 7
n_sample = 3000*2
traindataset = 100
simdataset = 100
deleteset = 15
ndataset = traindataset + simdataset + deleteset
n_group=100
group = np.linspace(0,1,n_group+1)
data = [[]]*category
train_n = [[]]*category
sim_n = [[]]*category
n_features=6
tempmaxtrain=np.zeros((category,n_features))
tempmaxsim=np.zeros((category,n_features))
N=5 #change N for better performance
weights=np.hanning(N)
weightsphase=np.hanning(5)
weightsfreq=np.hanning(25)
tao=2
print route
for j in range(0,category):
  data[j] = (ReadFile(route+filenamer[j]+filetype,n_sample*ndataset))
  #train
  d = np.mat([[data[j][i]+1j*data[j][i+1]] for i in range(0, n_sample*(traindataset+deleteset)-1, 2)])
  d = d.reshape((traindataset+deleteset,n_sample/2))
  d = d[deleteset:]
  
  freq = np.abs(np.fft.fft(d))
  freq = freq/float(freq.max())
  freqhanning =  np.zeros((np.size(freq,0),np.size(freq,1)))
  for i in range(0,traindataset):
    freqhanning[i]=np.convolve(weightsfreq/weightsfreq.sum(),freq[i],'same')
  freqhanning = freqhanning/float(freqhanning.max())
  
  distance = np.abs(d)
  distance = distance/distance.max()
  
  angle = np.angle(d)
  angle = (angle+np.pi)/(2*np.pi)
  
  histd, bin_edges = np.histogram(distance[0],bins=group)
  if filterflag:
    histd=np.convolve(weights/weights.sum(),histd,'same')
  countd = countpeak(histd,n_group)
  hista, bin_edges = np.histogram(angle[0],bins=group)
  if filteraflag:
    hista=np.convolve(weightsphase/weightsphase.sum(),hista,'same')
  counta = countpeak(hista,n_group)
  
  histd = histd/float(histd.max())
  hista = hista/float(hista.max())
  
  d1=np.array(d[0])
  d1=d1[0]
  d1=d1[0:len(d1)-tao]
  d2=np.array(d[0])
  d2=d2[0]
  d2=d2[tao:len(d2)]
  m21=np.mean(d1*np.conj(d2))
  m21_a=np.abs(m21)
  m21_p=(np.angle(m21)+np.pi)/(2*np.pi)
  
  temp = np.hstack((countd,100*np.var(hista),100*np.var(histd),100*np.var(freqhanning),100*m21_a,10*m21_p))#remove counta,np.mean(distance[0])
  for i in range(1,traindataset):
    histd, bin_edges = np.histogram(distance[i],bins=group)
    if filterflag:
      histd=np.convolve(weights/weights.sum(),histd,'same')
    countd = countpeak(histd,n_group)
    hista, bin_edges = np.histogram(angle[i],bins=group)
    if filteraflag:
      hista=np.convolve(weightsphase/weightsphase.sum(),hista,'same')
    counta = countpeak(hista,n_group)
    
    histd = histd/float(histd.max())
    hista = hista/float(hista.max())
    
    d1=np.array(d[i])
    d1=d1[0]
    d1=d1[0:len(d1)-tao]
    d2=np.array(d[i])
    d2=d2[0]
    d2=d2[tao:len(d2)]
    m21=np.mean(d1*np.conj(d2))
    m21_a=np.abs(m21)
    m21_p=(np.angle(m21)+np.pi)/(2*np.pi)
    temp = np.vstack((temp,np.hstack((countd,100*np.var(hista),100*np.var(histd),100*np.var(freqhanning),100*m21_a,10*m21_p))))#remove counta,np.mean(distance[i])
  tempmaxtrain[j]=np.max(temp,axis=0)
  train_n[j] = temp
    
  #sim
  d = np.mat([[data[j][i]+1j*data[j][i+1]] for i in range(n_sample*(traindataset+deleteset), n_sample*ndataset-1, 2)])
  d = d.reshape((simdataset,n_sample/2))
  
  freq = np.abs(np.fft.fft(d))
  freq = freq/float(freq.max())
  freqhanning =  np.zeros((np.size(freq,0),np.size(freq,1)))
  for i in range(0,simdataset):
    freqhanning[i]=np.convolve(weightsfreq/weightsfreq.sum(),freq[i],'same')
  freqhanning = freqhanning/float(freqhanning.max())
  
  distance = np.abs(d)
  distance = distance/distance.max()
  
  angle = np.angle(d)
  angle = (angle+np.pi)/(2*np.pi)
  
  histd, bin_edges = np.histogram(distance[0],bins=group)
  if filterflag:
    histd=np.convolve(weights/weights.sum(),histd,'same')
  countd = countpeak(histd,n_group)
  hista, bin_edges = np.histogram(angle[0],bins=group)
  if filteraflag:
    hista=np.convolve(weightsphase/weightsphase.sum(),hista,'same')
  counta = countpeak(hista,n_group)
  
  histd = histd/float(histd.max())
  hista = hista/float(hista.max())
  
  d1=np.array(d[0])
  d1=d1[0]
  d1=d1[0:len(d1)-tao]
  d2=np.array(d[0])
  d2=d2[0]
  d2=d2[tao:len(d2)]
  m21=np.mean(d1*np.conj(d2))
  m21_a=np.abs(m21)
  m21_p=(np.angle(m21)+np.pi)/(2*np.pi)
  
  temp = np.hstack((countd,100*np.var(hista),100*np.var(histd),100*np.var(freqhanning),100*m21_a,10*m21_p))#remove counta,np.mean(distance[0])
  for i in range(1,simdataset):
    histd, bin_edges = np.histogram(distance[i],bins=group)
    if filterflag:
      histd=np.convolve(weights/weights.sum(),histd,'same')
    countd = countpeak(histd,n_group)
    hista, bin_edges = np.histogram(angle[i],bins=group)
    if filteraflag:
      hista=np.convolve(weightsphase/weightsphase.sum(),hista,'same')
    counta = countpeak(hista,n_group)
  
    histd = histd/float(histd.max())
    hista = hista/float(hista.max())
    
    d1=np.array(d[i])
    d1=d1[0]
    d1=d1[0:len(d1)-tao]
    d2=np.array(d[i])
    d2=d2[0]
    d2=d2[tao:len(d2)]
    m21=np.mean(d1*np.conj(d2))
    m21_a=np.abs(m21)
    m21_p=(np.angle(m21)+np.pi)/(2*np.pi)

    temp = np.vstack((temp,np.hstack((countd,100*np.var(hista),100*np.var(histd),100*np.var(freqhanning),100*m21_a,10*m21_p))))#remove counta,np.mean(distance[i])
  tempmaxsim[j]=np.max(temp,axis=0)
  sim_n[j] = temp

# normalization
normflag=0
if normflag==1:
  print('normalization')
  trainmax=np.max(tempmaxtrain,axis=0)
  simmax=np.max(tempmaxsim,axis=0)
  trainmax = np.tile(trainmax,(traindataset,1))
  simmax = np.tile(simmax,(simdataset,1))
  for j in range(0,category):
    train_n[j] = train_n[j]/trainmax.astype('float32')*100
    sim_n[j] = sim_n[j]/simmax.astype('float32')*100
else:
  print('without norm')

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

print(np.sum(y==ytest)/(float(category)*simdataset))

for j in range(0,category):
  for i in range(0,category):
    print('y='+str(i)+',ytest='+str(j)+': ',np.sum(np.logical_and(y==i, ytest==j))/(float(simdataset)))

if 'simdata_agc' in route:
  xls_path = '/home/guyu/tongji/SOMNN+SVM/result/histcumulant_result_agc.xls'
elif 'simdata_withoutagc' in route:
  xls_path = '/home/guyu/tongji/SOMNN+SVM/result/histcumulant_result_withoutagc.xls'
else:
 xls_path = '/home/guyu/tongji/SOMNN+SVM/result/histcumulant_result.xls'
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
'''
for root,dirs,files in os.walk('/home/guyu/tongji/SOMNN+SVM/data'):
     for dir in dirs:
             print dir
             print(os.path.join(root,dir))
'''
