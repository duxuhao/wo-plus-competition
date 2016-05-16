# -*- coding: utf-8 -*-
"""
Created on Wed May 04 20:23:57 2016

@author: 21644336
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier # this method give the best, haven't try ANN
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import random
import codecs
from multiprocessing import Pool
import sys
from sklearn import preprocessing

#pool = Pool(4)

def predict(fea, df, t, t9):
    Un = df.columns == 'Blank'
    for f in Fea:
        '''        
        try:
            df[(f+'_y')] = df[(f+'_x')] - df[(f+'_y')]
            print(1)
        except:
            pass
        '''
        Un = Un | (df.columns == f)
        Un = Un | (df.columns == (f+'_x'))
        Un = Un | (df.columns == (f+'_y'))
    Un = Un & (df.columns != 'New_y')    
    clf = GradientBoostingClassifier()
    y = df[t].label
    X = df[t].ix[:,Un]
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size = 0.9, random_state = 1)
    clf.fit(X_train, y_train)
    re = 'Testing AUC: \t' + str(roc_auc_score(y_test,clf.predict_proba(X_test)[:,1]))  
    print re
    re =  'September AUC: \t' + str(roc_auc_score(df[t9].label,clf.predict_proba(df[t9].ix[:,Un])[:,1]))
    print re
    print(X.columns)
    print(clf.feature_importances_)
    return Un, clf

def add_pre(Un, clf, df,df1, name):
    df[name] = pd.Series(clf.predict_proba(df1.ix[:,Un])[:,1], index = df.index)
    return df
    
def writetofile(df, clf, Un):
    pre = df[df.Month > 201502]
    pre['score'] = pd.Series(np.round(clf.predict_proba(pre.ix[:,Un])[:,1],5), index = pre.index)
    new = pre[pre.Month == 201512][['IMSI','score']]
    new.columns = ['Idx','score']
    test = pd.read_csv('Test_IMSI_all_used.csv',encoding = 'utf-8',header = None)
    test.columns = ['Idx']
    pre = pd.merge(test, new,on='Idx', left_index=True,how='left')
    pre.to_csv('woplus_submit_sample.csv',encoding = 'utf-8', index = None)
    
def duplic(df, t, n):
    t.append('Month')
    t.append('IMSI')
    out = df.copy()
    for i in range(1, n):
        new = df[t]
        new.Month += i
	out = pd.merge(out, new, on=['IMSI','Month'], left_index = True, how = 'left')
    return out

df = pd.read_csv('Q2Cloud.csv',encoding = 'utf-8')
a = df.copy()
#Fea =['Trend_y','monthly_brand_counts','quarterly_attrition_rate','callave','networkave','flowave','smsave','apruave','Label', 'Price', 'Result_Quantity', 'compare1', 'ave', 'APRU', 'Gender', 'Network', 'Age', 'Flow', 'SMS', 'Call']
Fea = ['apruave','networkave','flowave','smsave','quarterly_attrition_rate','Brand_Counts','callave','Label', 'Price',  'compare1', 'ave', 'APRU', 'Gender', 'Network', 'Age', 'Flow', 'SMS','compare2','Call']
#Fea = ['New','Label', 'Price', 'Result_Quantity', 'compare1', 'Brand_Counts', 'ave', 'APRU', 'Gender', 'Network', 'Age', 'Flow', 'SMS', 'compare2', 'last_change', 'Call']
usedmonth = 1
a = duplic(df = a, t = Fea, n = usedmonth)
T = (a.Month == 201500)
for month in range(201500 + usedmonth,201507):
    T = (a.Month == month) | T
T9 = (a.Month == 201509)
Un, clf = predict(fea = Fea[:-2], df = a, t = T, t9 = T9)
df = add_pre(Un, clf,df,a,'New')
Fea = ['New','Label', 'Price', 'Result_Quantity', 'compare1', 'Brand_Counts', 'ave', 'APRU', 'Gender', 'Network', 'Age', 'Flow', 'SMS', 'compare2', 'last_change', 'Call']
a = df.copy()
usedmonth = 2
a = duplic(df = a, t = Fea, n = usedmonth)
T = (a.Month == 201500)
for month in range(201500 + usedmonth,201507):
    T = (a.Month == month) | T
T9 = (a.Month == 201509)
Un, clf = predict(fea = Fea[:-2], df = a, t = T, t9 = T9)


#writetofile(df = a, clf = clf, Un = Un)