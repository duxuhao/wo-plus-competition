# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 10:11:25 2016

@author: 21644336
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 20:18:18 2016

@author: Xuhao Du

"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier , RandomForestClassifier # this method give the best, haven't try ANN
from sklearn.cross_validation import train_test_split
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score


def read_label(n):
    name = 'label' + str(n) + '.csv'
    label=pd.read_csv(name,encoding='utf-8')
    return label
'''
-------------------------------------------------------------------------------------
this function is for training the data, clf_index = 0 use random forest; 1 use 
gradientboosting df is the dataframe, Tmon is the month we used to do the trainig, 
Testing_Percentage is the percetange data we use for testing
-------------------------------------------------------------------------------------
'''
def loop_train(df, Tmon, Test_Percentage, clf_index = 0):
    print ' ----- New loop ----- '
    X = df[Tmon]
    y = X.label
    Unused_Feature = (df.columns != 'Month') & (X.columns != 'Brand') & (X.columns != 'Model') & (X.columns != 'IMSI') & (X.columns != 'label')
    train = X.ix[:,Unused_Feature]
    X_train, X_test, y_train, y_test=train_test_split(train, y, test_size = Test_Percentage, random_state = np.random.randint(100000))
    clf =RandomForestClassifier()
    if clf_index == 1:   
        clf =GradientBoostingClassifier()
    clf.fit(X_train, y_train)
    auc = roc_auc_score(y_test,clf.predict_proba(X_test)[:,1])
    print 'This Loop AUC:           ' + str(auc)
    return clf
'''
-------------------------------------------------------------------------------------
this fuction is for output the prediction probability. df is the dataframe
clf is the classifier method
-------------------------------------------------------------------------------------
'''
def predict_prob(clf, df):
    Unused_Feature = (df.columns != 'Month') & (df.columns != 'Brand') & (df.columns != 'Model') & (df.columns != 'IMSI') & (df.columns != 'label')
    train = df.ix[:,Unused_Feature]
    prob = clf.predict_proba(train)
    return prob
'''
-------------------------------------------------------------------------------------
this fuction is for adding the last n month prediction probability. df is the dataframe
prob is the probability, n is the precious n month, name is the new features's name 
-------------------------------------------------------------------------------------
'''
def add_feature(prob, n, df, name):
    t1 = np.vstack((prob[-n:],prob[:-n]))
    df[name] = pd.Series(t1[:,1], index=df.index)
    return df
    
def predict_add_feature(clf, df, fearange, add_name = 0, begin = 0):
    prob = predict_prob(clf = clf, df = df)
    for i in range(begin,fearange-begin):
        df = add_feature(prob = prob, n = i, df = df, name = ('L' + str(fearange+1) + str(i+1) + str(add_name)))
    return df
'''
-------------------------------------------------------------------------------------
this fuction is choosing the used. df is the dataframe, select_range is which month
we need. 
-------------------------------------------------------------------------------------
'''  
def select_month(df, select_range):
    Tmon = np.zeros(len(df)).astype(bool)
    for mon in select_range:
        Tmon = Tmon | (df.Month == mon)
    return Tmon
    
def get_clf_with_selected_month(df, select_range, Test_Percentage, clf_index = 0):
    Tmon = select_month(df = df, select_range = select_range, )
    clf = loop_train(df=df, Tmon=Tmon, Test_Percentage=Test_Percentage, clf_index = clf_index)
    return clf
'''
-------------------------------------------------------------------------------------
this fuction is for merging the data 
-------------------------------------------------------------------------------------
''' 
def data_merge(df, label, Trend, test):
    df = pd.merge(df, label,on=['IMSI','Month'], left_index=False,how='left')
    df = df.ix[:, df.columns != 'Trend']
    df = pd.merge(df, Trend,on=['Month','Label'], left_index=False,how='left')
    df= pd.merge(df, test, on='IMSI', right_index=True,how='inner')  
    return df
'''
-------------------------------------------------------------------------------------
this fuction is for see the prediction AUC for verification
-------------------------------------------------------------------------------------
'''     
def verified(df, clf, month):
    print ' ---- Verification Set ----\n'
    T = (df.Month == month)
    auc2 = roc_auc_score(df[T].label,predict_prob(clf, df[T])[:,1])
    print 'The AUC for unkonwn Mynext three months is: ' + str(auc2)
'''
-------------------------------------------------------------------------------------
this fuction is for writting the corresponding data to file
-------------------------------------------------------------------------------------
'''  
def writetofile(df, clf, month, test, name = 'woplus_submit_sample.csv'):
    Unused_Feature = (df.columns != 'Month') & (df.columns != 'Brand') & (df.columns != 'Model') & (df.columns != 'IMSI') & (df.columns != 'label')
    train = df.ix[:,Unused_Feature]
    df['score'] = pd.Series(np.round(clf.predict_proba(train)[:,1],5), index=train.index)
    pre = df[df.Month == month]
    pre = pre[['IMSI','score']]
    pre.columns = ['Idx','score']
    test.columns = ['Idx','No']
    new = pd.merge(test['Idx'], pre,on='Idx', left_index=True,how='left')
    new.to_csv(name,encoding = 'utf-8', index = None)
'''
-------------------------------------------------------------------------------------
The main function, step_switch = 0, step equal to the Mynext gap, step = 1, step equal 
to 1
-------------------------------------------------------------------------------------
'''
def main(step_switch, Next_N_month, LoopNumber, used_previous_months, Test_Percentage,predict_month, origin, test, Trend):
    if used_previous_months > (Next_N_month - 1):
        print 'Not label to be used for the Mynext' + str(used_previous_months + 1) + ' month prediction'

    label = read_label(Next_N_month)
    df = data_merge(df = origin.copy(), label = label, Trend = Trend, test = test)
    new_set =df[['IMSI','Month','previous_label','previous_label2', 'Trend','label','Result_Quantity','labelcompare1_ave','No']]
    
    print '-' * 100
    print 'In this training, we use ' + str(used_previous_months+1) + " months' data to predict the Mynext 3 months' condiction"
    print 'Each training of the dataset  will be repeted for ' + str(LoopNumber + 1) + ' times\n'
    
    for Mynext in range(Next_N_month):
        print ' ---- Start the training for the Mynext ' + str(Mynext+1) + " months' label prediction----"
        label = read_label(Mynext+1)
        df = data_merge(df = origin.copy(), label = label, Trend = Trend, test = test)
        step = Mynext + 1
        add = np.random.randint(3)
        if step_switch:
            step = 0
            add = 0
        clf = get_clf_with_selected_month(df = df, select_range = np.arange(201501,predict_month-Mynext,step+1), Test_Percentage=Test_Percentage)  
        for ite in range(LoopNumber):
            df = predict_add_feature(clf = clf, df=df, fearange = used_previous_months + 1, add_name = ite, begin =1)
            clf = get_clf_with_selected_month(df = df, select_range = np.arange(201501 + used_previous_months + add,predict_month-Mynext,step+1), Test_Percentage=Test_Percentage, clf_index = (ite == LoopNumber-1))
        
        prob = predict_prob(clf = clf, df = df)
        for lo in range(Mynext+1):
            new_set = add_feature(prob = prob, n = lo, df = new_set, name =  ('L' + str(Mynext+1) + str(lo+1)))
            
        print '                 -- complete Mynext ' + str(Mynext+1) + " months' prediction --  "

    print '\n ---- Finish the prepared loop ----\n'
    print '\n ---- Start the Final loop ----\n'
    step = Next_N_month
    add = np.random.randint(3)
    if step_switch:
        step = 0
        add = 0
    verified(df = df, clf=clf, month = 201509)
    '''
    clf2 = get_clf_with_selected_month(df = new_set, select_range = np.arange(201501 + used_previous_months + add ,predict_month + 1 -Next_N_month, step + 1), Test_Percentage=Test_Percentage)  
    verified(df = new_set, clf=clf2, month = 201509)
    new_set = predict_add_feature(clf = clf2, df = new_set, fearange = used_previous_months + 1, begin = 1)    
    clf2 = get_clf_with_selected_month(df = new_set, select_range = np.arange(201501 + used_previous_months + add ,predict_month + 1 -Next_N_month, step + 1), Test_Percentage=Test_Percentage)  
    verified(df = new_set, clf=clf2, month = 201509)
    new_set = predict_add_feature(clf = clf2, df = new_set, fearange = used_previous_months + 1, begin = 1)    
    '''
    clf3 = get_clf_with_selected_month(df = new_set, select_range = np.arange(201501 + used_previous_months + add ,predict_month + 1 -Next_N_month, step + 1), Test_Percentage=Test_Percentage, clf_index = 1)
    verified(df = new_set, clf=clf3, month = 201509)  
    writetofile(df = new_set, clf = clf3, test = test, month = predict_month)
    return clf, clf3, new_set
      
pool = Pool(8)
origin=pd.read_csv('Q2_merge_all.csv',encoding='utf-8') # this used dataset with the basic privided parameter
test = pd.read_csv('Test_IMSI_all_used.csv',encoding = 'utf-8')
Trend = pd.read_csv('Trend_new.csv',encoding = 'utf-8')
final_clf1, final_clf3, dataset= main(step_switch = 1, Next_N_month = 3, LoopNumber = 2, used_previous_months = 2, Test_Percentage = 0.5, predict_month = 201509, origin = origin, test = test, Trend = Trend)