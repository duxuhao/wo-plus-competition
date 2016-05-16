# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 16:13:40 2016

@author: 21644336
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier # this method give the best, haven't try ANN
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score

origin=pd.read_csv('Q2_merge_all.csv',encoding='utf-8')
total = pd.read_csv('Total_change.csv',encoding = 'utf-8')
last = pd.read_csv('Month_to_last_change.csv',encoding = 'utf-8')
last.columns = ['IMSI','Month','Last_month_change']
test = pd.read_csv('Test_IMSI_all_used.csv',encoding = 'utf-8')
new = pd.merge(origin.copy(), test, on = 'IMSI', right_index = True, how = 'inner')
new = pd.merge(new, total, on = 'IMSI', left_index = True, how = 'left')
new = pd.merge(new, last, on=['IMSI', 'Month'], left_index = True, how = 'left')
label = pd.read_csv('change_phone_3_month.csv',encoding = 'utf-8' )
Unew = pd.merge(new.copy(), label, on=['IMSI', 'Month'], left_index = True, how = 'left')
Unew.Month += -201500
Feature = ['IMSI','Month','Result_Quantity','Price','Trend','Total','Last_month_change','label','Age','APRU','Call','Flow','SMS']
Used = Unew[Feature]
#T = (Used.Month == 1) |(Used.Month == 2) |(Used.Month == 3) |(Used.Month == 4) |(Used.Month == 5) |(Used.Month == 6) |(Used.Month == 7) |(Used.Month == 8) |(Used.Month == 9)
T = (Used.Month == 1) |(Used.Month == 2) |(Used.Month == 3) |(Used.Month == 4) |(Used.Month == 5) |(Used.Month == 6)
Un = (Used.columns != 'IMSI') & (Used.columns != 'label')
X_train, X_test, y_train, y_test=train_test_split(Used[T].ix[:, Un], Used[T].label, test_size = 0.5, random_state = np.random.randint(100000))
clf =RandomForestClassifier()
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
print(clf.feature_importances_)
print(roc_auc_score(Used[Used.Month == 9].label, clf.predict_proba(Used[Used.Month == 9].ix[:, Un])[:,1]))
aa = clf.predict_proba(Used.ix[:, Un])
Used['Pre1'] = pd.Series(np.vstack((aa[-0:],aa[:-0]))[:,1], index = Used.index)
Used['Pre2'] = pd.Series(np.vstack((aa[-1:],aa[:-1]))[:,1], index = Used.index)
Used['Pre3'] = pd.Series(np.vstack((aa[-2:],aa[:-2]))[:,1], index = Used.index)
Un = (Used.columns != 'IMSI') & (Used.columns != 'label') & (Used.columns != 'Month')
X_train, X_test, y_train, y_test=train_test_split(Used[T].ix[:, Un], Used[T].label, test_size = 0.5, random_state = np.random.randint(100000))
clf =GradientBoostingClassifier()
clf.fit(X_train, y_train)
print(roc_auc_score(Used[Used.Month == 9].label, clf.predict_proba(Used[Used.Month == 9].ix[:, Un])[:,1]))
T = Used.Month == 12
a = Used.ix[:, Un].copy()
Used['score'] = pd.Series(np.round(clf.predict_proba(a)[:,1],5), index = Used.index)
score_file = Used[T]
score_file1 = score_file[['IMSI','score']]
score_file1.columns = ['Idx','score']
test.columns = ['Idx']
new = pd.merge(test, score_file1,on='Idx', left_index=True,how='left')
new.to_csv('woplus_submit_sample.csv',encoding = 'utf-8', index = None)