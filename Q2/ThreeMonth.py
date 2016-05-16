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
from sklearn import preprocessing

def predict(fea1,fea2, df, t, t9):
    n = 0
    weight = [0.73,0.27]
    tave = np.zeros(len(df[t9]))
    y = df[t].label
    X_1 = df[t]
    df9 = df[t9]
    for fea in [fea1,fea2]:
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
        re = 'Testing AUC: \t' + str(roc_auc_score(y_test,clf.predict_proba(min_max_scaler.transform(X_test))[:,1]))
        print re
        t = clf.predict_proba(min_max_scaler.fit_transform(df9.ix[:,Un]))[:,1]
        re =  'September AUC: \t' + str(roc_auc_score(df9.label,t))
        print re
        tave = t * weight[n] + tave
        n += 1
        
    
    print '-' * 30
    print(weight)
    print 'Total AUC'
    re =  'September AUC: \t' + str(roc_auc_score(df9.label,tave))
    print re
    return Un, clf

    
def writetofile(df, clf, Un, test):
    pre = df[df.Month > 201501]
    pre['score'] = pd.Series(np.round(clf.predict_proba(pre.ix[:,Un])[:,1],5), index = pre.index)
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
df =pd.read_csv('Q2Cloud.csv',encoding = 'utf-8')
a = df.copy()
Fea1 = ['quarterly_attrition_rate','callave','Label', 'Price', 'ave', 'APRU', 'Gender', 'Network', 'Age', 'Flow', 'SMS', 'compare2', 'last_change', 'Call', 'Result_Quantity', 'compare1', 'Brand_Counts','Trend_x']
Fea2 = ['Label', 'Price', 'Result_Quantity', 'compare1', 'Brand_Counts', 'ave', 'APRU', 'Gender', 'Network', 'Age', 'Flow', 'SMS', 'compare2', 'last_change', 'Call'] 
a = duplic(df = a, t = Fea1)
T = (a.Month == 201503) |(a.Month == 201503) |(a.Month == 201504) |(a.Month == 201505) |(a.Month == 201506)#|(a.Month == 201507) |(a.Month == 201508) |(a.Month == 201509)
T9 = (a.Month == 201509)

Un, clf = predict(fea1 = Fea1[:-3], fea2 = Fea2, df = a, t = T, t9 = T9)

#writetofile(df = a, clf = clf, Un = Un, test = test)
