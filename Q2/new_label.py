# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 11:25:53 2016

@author: 21644336
"""
import pandas as pd
import numpy as np
origin=pd.read_csv('Q2_merge_all.csv',encoding='utf-8')
label1 = pd.read_csv('all_labels.csv')
#label1.Label_Last1 = 0
label.columns = ['IMSI','Label_Last1','Label_Last2','Label_Last3','Label_Last4','Label_Last5','Label_Last6','Label_Last7','Label_Last8','Label_Last9','Label_Last10','Label_Last11','Label_Last12']
T = label[['IMSI']].copy()
n = 3 #the phone in next n months
for i in range(12):
    name1 = str(201501 +i)
    T[name1] = pd.Series(np.sum(label.ix[:,i+2:i+5],axis = 1).astype(bool).astype(int), index = T.index)

filename = 'change_phone_' + str(n) + '_month.csv'


frame1 = []
for i in range(12):
    variable_name = str(201501 + i)
    frame1.append(pd.DataFrame({'Month':np.ones(len(T))*i + 201501,'IMSI':T.IMSI, 'label': T[variable_name]}))

phone_change = pd.concat(frame1)
phone_change.to_csv(filename,encoding = 'utf-8',index = None)

# for the month to last change phone
a = np.sum(label1.ix[:,1:12],axis = 1)
ave = a
ave[ave == 0] = sum(a)/360698
T = label1[['IMSI']].copy()
T['201501'] = pd.Series(np.zeros(len(T)), index = T.index)
for i in range(11):
    name1 = str(201502 + i)
    name2 =str(201501 + i)
    valuename = 'Label_Last' + str(i+2)
    a2 = 1 - label1[valuename]
    a2[a2 == 1] = T[name2][a2 == 1] + 1
    T[name1] = pd.Series(a2, index = T.index)

#T.to_csv('Month_to_last_change.csv',encoding = 'utf-8')
frame1 = []
for i in range(12):
    variable_name = str(201501 + i)
    frame1.append(pd.DataFrame({'Month':np.ones(len(T))*i + 201501,'IMSI':T.IMSI, 'last_change': T[variable_name]}))

phone_change = pd.concat(frame1)
phone_change.to_csv('Month_to_last_change.csv',encoding = 'utf-8',index = None)

T = label1[['IMSI']].copy()
T['Total'] = pd.Series(a, index = T.index)
T.to_csv('Total_change.csv',encoding = 'utf-8',index = None)