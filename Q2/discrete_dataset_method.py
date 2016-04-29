# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 20:18:18 2016

@author: 21644336
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 19:48:55 2016

@author: 21644336
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 20:34:10 2016

@author: 21644336
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier , RandomForestClassifier # this method give the best, haven't try ANN
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.cross_validation import train_test_split
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

pool = Pool(8)
origin=pd.read_csv('Q2_merge_all.csv',encoding='utf-8') # this used dataset with the basic privided parameter
df = origin.copy()
test = pd.read_csv('Test_IMSI_all_used.csv',encoding = 'utf-8',header = None)
Test_Percentage = 0.3
test.columns = ['IMSI']
Trend = pd.read_csv('Trend_new.csv',encoding = 'utf-8')
label=pd.read_csv('label3.csv',encoding='utf-8')
df = pd.merge(df, label,on=['IMSI','Month'], left_index=False,how='left')
df = df.ix[:, df.columns != 'Trend']
df = pd.merge(df, Trend,on=['Month','Label'], left_index=False,how='left')
df= pd.merge(df, test, on='IMSI', right_index=True,how='inner')

new_set =df[['IMSI','Month','previous_label','previous_label2', 'Trend','label','Result_Quantity','labelcompare1_ave']]
LoopNumber =2 #loop times
N = 2 #use previous N months predict

print '\n'
print '-' * 100
print 'In this training, we use ' + str(N+1) + "months' data to predict the next 3 months' condiction"
print 'Each training of the dataset  will be repeted for ' + str(LoopNumber + 1) + ' times'
print '\n'

for next in range(N+1):
    print ' ---- Start the training for the next ' + str(next+1) + " months' label prediction----"
    name='label'+str(next+1) +'.csv'
    label=pd.read_csv(name,encoding='utf-8')
    df = origin.copy()
    df = pd.merge(df, label,on=['IMSI','Month'], left_index=False,how='left')
    df = df.ix[:, df.columns != 'Trend']
    df = pd.merge(df, Trend,on=['Month','Label'], left_index=False,how='left')
    df= pd.merge(df, test, on='IMSI', right_index=True,how='inner')    
    #zero round
    
    Tmon = np.zeros(len(df)).astype(bool)
    for mon in np.arange(201501,201509-next,next+2):
        Tmon = Tmon | (df.Month == mon)
    

    X = df[Tmon]
    y = X.label
    #delete features that cannot be used
    Unused_Feature = (df.columns != 'Month') & (X.columns != 'Brand') & (X.columns != 'Model') & (X.columns != 'IMSI') & (X.columns != 'label')
    train = X.ix[:,Unused_Feature]
    X_train, X_test, y_train, y_test=train_test_split(train, y, test_size = Test_Percentage, random_state = np.random.randint(100000))
    clf =RandomForestClassifier()
    clf.fit(X_train, y_train)
    auc = roc_auc_score(y_test,clf.predict_proba(X_test)[:,1])
    print '0 Loop AUC:           ' + str(auc)
    for ite in range(LoopNumber):
        
        train = df.ix[:,Unused_Feature]
        prob = clf.predict_proba(train)
        
        for i in range(N):
            dfname2 = 'L' + str(ite * N + i + 1)
            t1 = np.vstack((prob[-i-1:],prob[:-i-1]))
            df[dfname2] = pd.Series(t1[:,1], index=df.index)
        Tmon = np.zeros(len(df)).astype(bool)
        for mon in np.arange(201501+N + np.random.randint(3) ,201509-next,next+2):
            Tmon = Tmon | (df.Month == mon)
                
        X = df[Tmon] 
        y = X.label
        Unused_Feature = (X.columns != 'Month') & (X.columns != 'Brand') & (X.columns != 'Model') & (X.columns != 'IMSI') & (X.columns != 'label')
        train = X.ix[:,Unused_Feature]
        X_train, X_test, y_train, y_test=train_test_split(train, y, test_size = Test_Percentage, random_state = np.random.randint(100000))
        clf =RandomForestClassifier()    
        if (ite == LoopNumber-1):
               clf = GradientBoostingClassifier()
        clf.fit(X_train, y_train)
        auc = roc_auc_score(y_test,clf.predict_proba(X_test)[:,1])
        print str(ite+1) + ' Loop AUC:           ' + str(auc)
        
    X = df
    Unused_Feature = (X.columns != 'Month') & (X.columns != 'Brand') & (X.columns != 'Model') & (X.columns != 'IMSI') & (X.columns != 'label')
    train = X.ix[:,Unused_Feature]
    prob= clf.predict_proba(train)
    for lo in range(next+1):
        t1 = np.vstack((prob[-lo:],prob[:-lo]))
        name = str(next+1) + str(lo+1)
        new_set[name] = pd.Series(t1[:,1], index=new_set.index)
    print '                 -- complete next ' + str(next+1) + " months' prediction --  "
print '\n'
print ' ---- Finish the prepared loop ----'
print '\n'

Tmon = np.zeros(len(df)).astype(bool)
for mon in np.arange(201501+N + np.random.randint(3) ,201509-N,N+2):
    Tmon = Tmon | (df.Month == mon)  
clf2 = loop_train(new_set, Tmon, Test_Percentage)
Unused_Feature = (new_set.columns != 'Month') & (new_set.columns != 'IMSI') & (new_set.columns != 'label')
X = new_set[Tmon]
train = X.ix[:, Unused_Feature]
y = X.label
X_train, X_test, y_train, y_test=train_test_split(train, y, test_size = Test_Percentage, random_state = np.random.randint(100000))
clf = RandomForest()
clf.fit(X_train, y_train)
prob = clf.predict_proba(new_set.ix[:, Unused_Feature])
t1 = np.vstack((prob[-1:],prob[:-1]))
new_set['last'] = pd.Series(t1[:,1], index=new_set.index)
#clf2 = loop_train(new_set, Tmon, Test_Percentage, clf_index = 1)

Unused_Feature = (new_set.columns != 'Month') & (new_set.columns != 'IMSI') & (new_set.columns != 'label')
X = new_set[Tmon]
train = X.ix[:, Unused_Feature]
y = X.label
X_train, X_test, y_train, y_test=train_test_split(train, y, test_size = Test_Percentage, random_state = np.random.randint(100000))
clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)
auc = roc_auc_score(y_test,clf.predict_proba(X_test)[:,1])

print 'Final test set AUC: ' + str(auc)
print '\n'
print ' ---- Verification Set ----'
print '\n'
T = (new_set.Month == 201509)
auc2 = roc_auc_score(new_set[T].label,clf.predict_proba(new_set[T].ix[:, Unused_Feature])[:,1])
print 'The AUC for unkonwn next three months is: ' + str(auc2)

print '  ---- Write to the file ---- '
Unused_Feature = (new_set.columns != 'Month') & (new_set.columns != 'Brand') & (new_set.columns != 'Model') & (new_set.columns != 'IMSI') & (new_set.columns != 'label')
train = new_set.ix[:,Unused_Feature]
new_set['score'] = pd.Series(np.round(clf.predict_proba(train)[:,1],5), index=train.index)
pre = new_set[new_set.Month == 201512]
pre = pre[['IMSI','score']]
pre.columns = ['Idx','score']
test.columns = ['Idx']
new = pd.merge(test, pre,on='Idx', left_index=True,how='left' )
#new.to_csv('woplus_submit_sample.csv',encoding = 'utf-8', index = None)

'''
print '  ---- Write to the file ---- '
Unused_Feature = (df.columns != 'Month') & (df.columns != 'Brand') & (df.columns != 'Model') & (df.columns != 'IMSI')
train = df.ix[:,Unused_Feature]
df['score'] = pd.Series(clf.predict_proba(train)[:,1], index=train.index)
pre = df[df.Month == 201512]
pre = pre[['IMSI','score']]
pre.columns = ['Idx','score']
test.columns = ['Idx']
new = pd.merge(test, pre,on='Idx', left_index=True,how='left' )
new.to_csv('Test_IMSI_all.csv',encoding = 'utf-8', index = None)
'''