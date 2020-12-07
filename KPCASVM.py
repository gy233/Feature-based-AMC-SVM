import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA, KernelPCA
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

#route = '/home/guyu/tongji/SOMNN+SVM/data/n316mf0m/'
route = '/home/guyu/tongji/SOMNN+SVM/simdata_agc/n3160mf200m/'
route = '/home/guyu/tongji/SOMNN+SVM/simdata_withoutagc/n1000mf100m/'
#route = '/home/guyu/tongji/SOMNN+SVM/data/real_without_agc/'
#route = '/home/guyu/tongji/SOMNN+SVM/data/real_agc/'
filenamer = ['BPSK','QPSK','QAM16','QAM64','QAM256','GMSK','OFDM']
filetype = '.bin'
color = ['r','g','b','k','m','y','orange']
PCAflag = 1 # 0:KPCA    1:PCA
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
    for i in range(1,3):
      plt.subplot(2,2,i)
      plt.bar(groupplot,np.array(train_n[j][i]),width=1/float(n_group),align='edge')
      plt.title('train')
    for i in range(3,5):
      plt.subplot(2,2,i)
      plt.bar(groupplot,np.array(sim_n[j][i]),width=1/float(n_group),align='edge')
      plt.title('sim')
    plt.suptitle(filenamer[j])  
plt.show()
print('train_n.shape',len(train_n),len(train_n[0]))
train = train_n[0]
for j in range(1,category):
  train = np.vstack((train,train_n[j]))
sim = sim_n[0]
for j in range(1,category):
  sim = np.vstack((sim,sim_n[j]))
#PCA
components = 0.99  #canshu 
if PCAflag==1:
  pca = PCA(n_components=components,svd_solver='full')
  pca.fit(train)
  train_new=pca.transform(train)
  sim_new = pca.transform(sim)
  print('pca.explained_variance_ratio_',pca.explained_variance_ratio_)
  print('sum(pca.explained_variance_ratio_)',sum(pca.explained_variance_ratio_))
  print(pca.singular_values_)  
else:
  kpca = KernelPCA(n_components=components, kernel="rbf", fit_inverse_transform=True)
  kpca.fit(train)
  train_new=kpca.transform(train)
  sim_new = kpca.transform(sim)
  print('pca.explained_variance_ratio_',pca.explained_variance_ratio_)
  print('sum(pca.explained_variance_ratio_)',sum(pca.explained_variance_ratio_))
  print(pca.singular_values_) 
print('train.shape',train.shape) 
print('train_new.shape',train_new.shape) 
 
if plotflag==1:  
  plt.figure(figsize=(10, 8))
  for i in range(0,category):
    plt.subplot(1,2,1)
    plt.scatter(train_new[i*traindataset:(i+1)*traindataset, 0], train_new[i*traindataset:(i+1)*traindataset, 1],marker='o',c=color[i])
    plt.title('train')
    plt.subplot(1,2,2)
    plt.scatter(sim_new[i*simdataset:(i+1)*simdataset, 0], sim_new[i*simdataset:(i+1)*simdataset, 1],marker='o',c=color[i])
    plt.title('test')
  plt.show()
xtrain = train_new
xtest = sim_new

#SVM
ytrain = np.zeros((traindataset,1))
for j in range(1,category):
  ytrain = np.vstack((ytrain,j*np.ones((traindataset,1))))
ytrain = ytrain.ravel()
        
classifier = SVC(kernel='rbf',gamma='scale', C=1,decision_function_shape='ovo').fit(xtrain,ytrain)
y = classifier.predict(xtest)
ytest = np.zeros((1,simdataset))
for j in range(1,category):
  ytest = np.hstack((ytest,j*np.ones((1,simdataset))))
ytest = ytest.ravel()

print(np.sum(y==ytest)/(float(category)*simdataset))

for j in range(0,category):
  for i in range(0,category):
    print('y='+str(i)+',ytest='+str(j)+': ',np.sum(np.logical_and(y==i, ytest==j))/(float(simdataset)))

if 'simdata_agc' in route:
  xls_path = '/home/guyu/tongji/KPCA+SVM/result_agc.xls'
elif 'simdata_withoutagc' in route:
  xls_path = '/home/guyu/tongji/KPCA+SVM/result_withoutagc.xls'
else:
 xls_path = '/home/guyu/tongji/KPCA+SVM/result_99.xls'
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
  sheet.write(category+1,0,'n_features')
  sheet.write(category+1,1,train_new.shape[1])
  workbook.save(xls_path)
  print(sheetname+' has been added successfully')
else:
  print(sheetname+' exists in the file')
