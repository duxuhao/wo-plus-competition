# -*- coding: utf-8 -*-
"""
Created on Sat May 14 16:54:15 2016

@author: 21644336
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 04 20:23:57 2016

@author: 21644336
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier # this method give the best, haven't try ANN
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn import preprocessing
import scipy

def predict(fea1,fea2, df, t, t9):
    y = df[t].label
    X_1 = df[t]
    for fea in [fea2]:
        Un = df.columns == 'Blank'
        for f in fea:
            Un = Un | (df.columns == f)
            Un = Un | (df.columns == (f+'_x'))
            Un = Un | (df.columns == (f+'_y'))
        Un = Un & (df.columns != 'quarterly_attrition_rate_y')
        clf = GradientBoostingClassifier()
        X = X_1.ix[:,Un]
        X_train, X_test, y_train, y_test=train_test_split(X, y, test_size = 0.9, random_state = 1)  
        min_max_scaler = preprocessing.MinMaxScaler()
        clf.fit(min_max_scaler.fit_transform(X_train), y_train)

    return Un, clf
    
def writetofile(df, clf, Un, test):
    pre = df[df.Month > 201501]
    min_max_scaler = preprocessing.MinMaxScaler()
    pre['score'] = pd.Series(np.round(clf.predict_proba(min_max_scaler.fit_transform(pre.ix[:,Un]))[:,1],5), index = pre.index)
    new = pre[pre.Month == 201512][['IMSI','score']]
    new.columns = ['Idx','score']
    test.columns = ['Idx']
    pre = pd.merge(test, new,on='Idx', left_index=True,how='left')
    #pre.to_csv('woplus_submit_sample.csv',encoding = 'utf-8', index = None)
    return pre
    
def duplic(df, t):
    t.append('IMSI')
    t.append('Month')
    new = df[t]
    new.Month += 1
    new = pd.merge(df, new, on=['IMSI','Month'], left_index = True, how = 'left')
    return new

test = pd.read_csv('Test_IMSI_all_used.csv',encoding = 'utf-8')
#df =pd.read_csv('Q2Cloud.csv',encoding = 'utf-8')
df =pd.read_csv('submit_result/657Data.csv',encoding = 'utf-8')
a = df.copy()
#Fea1 = ['system','Trend_x','quarterly_attrition_rate','callave','Label', 'Price', 'Result_Quantity', 'compare1', 'Brand_Counts', 'ave', 'APRU', 'Gender', 'Network', 'Age', 'Flow', 'SMS', 'compare2', 'last_change', 'Call']
Fea2 = ['Trend_x','Label', 'Price', 'Result_Quantity', 'compare1', 'Brand_Counts', 'ave', 'APRU', 'Gender', 'Network', 'Age', 'Flow', 'SMS', 'compare2', 'last_change', 'Call'] 
a = duplic(df = a, t = Fea2)
T = (a.Month == 201502) |(a.Month == 201503) |(a.Month == 201504) |(a.Month == 201505) |(a.Month == 201506)|(a.Month == 201507) |(a.Month == 201508) |(a.Month == 201509)
T9 = (a.Month == 201509)

Un, clf = predict(fea1 = Fea2[:-2], fea2 = Fea2[:-2], df = a, t = T, t9 = T9)

pre = writetofile(df = a, clf = clf, Un = Un, test = test)
A = pd.read_csv('submit_result/657CADV20160505.csv')
print(scipy.stats.pearsonr(pre.score, A.score))
'''
['Label', 'Price', 'Result_Quantity', 'compare1', 'Brand_Counts', 'ave', 'APRU', 'Gender', 'Network', 'Age', 'Flow', 'SMS', 'compare2', 'last_change', 'Call']
Testing AUC:    0.683709829836
September AUC:  0.659918850208
'''
