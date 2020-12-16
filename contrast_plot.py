import os,xlrd,xlwt
import matplotlib.pyplot as plt
import math

def get_plot_data(sheets,wb,line='ro-',lb='PCA+SVM',posh='center',posv='bottom'):
  x=[]
  y=[]
  for i in range(0,len(sheets)):
    if sheets[i] in wb.sheet_names():
      sheet = wb.sheet_by_name(sheet_name=sheets[i])
      x=x+[20*(3-math.log10(int(sheets[i].split('m')[0].split('n')[1])))]
      y=y+[sheet.cell_value(rowx=9, colx=1)*100]
  plt.plot(x,y,line,label=lb)
  for x,y in zip(x,y):
    plt.text(x, y , '%.2f' % y, ha=posh, va=posv,color=line[0])

sheets0 = ['n10mf0m','n100mf0m','n316mf0m','n1000mf0m','n3160mf0m']
sheets10 = ['n10mf10m','n100mf10m','n316mf10m','n1000mf10m','n3160mf10m']
sheets100 = ['n10mf100m','n100mf100m','n316mf100m','n1000mf100m','n3160mf100m']
sheets200 = ['n10mf200m','n100mf200m','n316mf200m','n1000mf200m','n3160mf200m']

wb1 = xlrd.open_workbook('/home/guyu/tongji/KPCA+SVM/result_agc.xls') 
wb2 = xlrd.open_workbook('/home/guyu/tongji/SOMNN+SVM/result/histcount_result_agc.xls') 
wb3 = xlrd.open_workbook('/home/guyu/tongji/SOMNN+SVM/result/histcumulant_result_agc.xls') 
wb4 = xlrd.open_workbook('/home/guyu/tongji/SOMNN+SVM/result/histNN_result_agc.xls') 

plt.figure()
get_plot_data(sheets0,wb1)
get_plot_data(sheets0,wb2,'bo-','features+SVM','right','center') 
get_plot_data(sheets0,wb3,'go-','features+cumulant+SVM','left','top') 
get_plot_data(sheets0,wb4,'ko-','nueral network','left','center') 
plt.xlim(-20,50)
plt.ylim(20,110)
plt.xlabel('SNR (dB)')
plt.ylabel('Accuracy (%)')
plt.legend(loc='lower right')
plt.grid(True)
plt.title('frequency offset = 0')

plt.figure()
get_plot_data(sheets10,wb1)
get_plot_data(sheets10,wb2,'bo-','features+SVM','right','top')
get_plot_data(sheets10,wb3,'go-','features+cumulant+SVM','left','top')
get_plot_data(sheets0,wb4,'ko-','nueral network','left','center') 
plt.xlim(-20,50)
plt.ylim(20,110)
plt.xlabel('SNR (dB)')
plt.ylabel('Accuracy (%)')
plt.legend(loc='lower right')
plt.grid(True)
plt.title('frequency offset = 0.01 fs')

plt.figure()
get_plot_data(sheets100,wb1)
get_plot_data(sheets100,wb2,'bo-','features+SVM','right','top') 
get_plot_data(sheets100,wb3,'go-','features+cumulant+SVM','left','top')
get_plot_data(sheets0,wb4,'ko-','nueral network','left','center') 
plt.xlim(-20,50)
plt.ylim(20,110)
plt.xlabel('SNR (dB)')
plt.ylabel('Accuracy (%)')
plt.legend(loc='lower right')
plt.grid(True)
plt.title('frequency offset = 0.1 fs')

plt.figure()
get_plot_data(sheets200,wb1)
get_plot_data(sheets200,wb2,'bo-','features+SVM','right','top') 
get_plot_data(sheets200,wb3,'go-','features+cumulant+SVM','left','top')
get_plot_data(sheets0,wb4,'ko-','nueral network','left','center') 
plt.xlim(-20,50)
plt.ylim(20,110)
plt.xlabel('SNR (dB)')
plt.ylabel('Accuracy (%)')
plt.legend(loc='lower right')
plt.grid(True)
plt.title('frequency offset = 0.2 fs')

#real_agc#
wb1 = xlrd.open_workbook('/home/guyu/tongji/KPCA+SVM/result_99.xls') 
wb2 = xlrd.open_workbook('/home/guyu/tongji/SOMNN+SVM/result/histNN_result.xls') 
wb3 = xlrd.open_workbook('/home/guyu/tongji/SOMNN+SVM/result/histcumulant_result.xls') 
wb4 = xlrd.open_workbook('/home/guyu/tongji/SOMNN+SVM/result/histcount_result.xls') 
wbs=[wb1,wb2,wb3,wb4]
name=('PCA+SVM','nueral network','features+cumulant+SVM','features+SVM')
x = []
y=[]
label = []
for index,wb in enumerate(wbs):
  if 'real_agc' in wb.sheet_names():
    sheet = wb.sheet_by_name(sheet_name='real_agc')
    label=label+[name[index]]
    y=y+[sheet.cell_value(rowx=9, colx=1)*100]
    x=x+[len(y)]
plt.figure()
plt.bar(x,y,label='Accuracy(%)',align='center',width=0.5,tick_label=label)
for a, b in zip(x, y):
  plt.text(a, b + 0.05, '%.2f' % b, ha='center', va='bottom')
plt.xlim(0,len(y)+1)
plt.ylim(0,110)
plt.legend()
plt.title('Accuracy of different methods in real world')
plt.show()


