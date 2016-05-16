# -*- coding: utf-8 -*-
"""
Created on Tue May 03 02:33:22 2016

@author: 21644336
"""
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier # this method give the best, haven't try ANN
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
import random
import codecs
#Feature
'''
IMSI, Month, Network, Gender, Age, APRU, Brand, Model, Flow, Call, SMS, Result_Quantity
Price, labelcompare1_ave, previous_label, previous_label2, Label, Trend_x, Trend_y, Total
last_change_x, compare1, compare2, ave, label, system, last_change_y, Brand_Counts
'''
'''
df =pd.read_csv('Q2New.csv',encoding = 'utf-8')
'''
#a = df.copy()

T1_6 = (a.Month == 201502) |(a.Month == 201502) |(a.Month == 201503) |(a.Month == 201504) |(a.Month == 201505) |(a.Month == 201506)
T9 = (a.Month == 201509)
y = a[T1_6].label

filew = codecs.open('log5.txt','a','utf-8')
#Fea = ['compare2','compare1','Brand_Counts','last_change','Price','Trend_x','Trend_y','Result_Quantity','system','ave','Label','Call','SMS','Age','Network','Flow','APRU','Gender'] #this combination have 0.636 T9 AUC
#Fea = ['compare2','compare1','Brand_Counts_x','last_change_x','Price_x','Trend_x_x','Result_Quantity_x','system_x','ave','Label_x','Call_x','SMS_x','Age','Network_x','Flow_x','Brand_Counts_y','last_change_y','Price_y','Trend_x_y','Result_Quantity_y','system_y','Label_y','Call_y','SMS_y','Network_y','Flow_y']
Fea = [u'Network_x', u'Gender_x', u'Age_x', u'APRU_x',
       u'Flow_x', u'Call_x', u'SMS_x',
       u'Label_x', u'Trend_x_x', u'Trend_y',
       u'ave_x', u'system_x', u'Brand_Counts_x',
       u'compare1_x', u'compare2_x', u'compare2_y', u'compare1_y',
       u'Brand_Counts_y', u'last_change_y', u'Price_y', u'Trend_x_y',
       u'Result_Quantity_y', u'system_y', u'ave_y', u'Label_y', u'Call_y',
       u'SMS_y', u'Age_y', u'Network_y', u'Flow_y', u'APRU_y', u'Gender_y','brand_increase_x','monthly_brand_counts_x','monthly_brand_counts_y']

if 1:
    for t in range(14,35):
        for n in range(20):
            try:
                Un = a.columns == 'Result_Quantity_x'
                Un = Un | (a.columns == 'last_change_x')
                Un = Un | (a.columns == 'Price_x')
                FeaSelect = random.sample(set(Fea), t)
                for f in FeaSelect:
                    Un = Un | (a.columns == f)
                
                X = a[T1_6].ix[:,Un]
                
                X_train, X_test, y_train, y_test=train_test_split(X, y, test_size = 0.7, random_state = 1)
                
                clf = GradientBoostingClassifier()
                
                clf.fit(X_train, y_train)
                print '-' * 40
                filew.write('-' * 40)
                filew.write('\n')
                print 'Features \t imortance: \n'
                filew.write('Features \t imortance: \n')
                for i, f in enumerate(X_train.columns):
                    if len(f) < 6:
                        re = f + '  \t\t' + str(round(clf.feature_importances_[i], 5))
                        print re
                    elif len(f)>14:
                        re = f + ' ' + str(round(clf.feature_importances_[i], 5))
                        print re
                    else:
                        re = f + '  \t' + str(round(clf.feature_importances_[i], 5))
                        print re
                    filew.write(re)
                    filew.write('\n')
                    
                print '-' * 40
                filew.write('-' * 40)
                filew.write('\n')
                re = 'Testing AUC: \t' + str(roc_auc_score(y_test,clf.predict_proba(X_test)[:,1]))
                print re
                filew.write(re)
                filew.write('\n')
                re =  'September AUC: \t' + str(roc_auc_score(a[T9].label,clf.predict_proba(a[T9].ix[:,Un])[:,1]))
                print re
                filew.write(re)
                filew.write('\n')
            except:
                  pass  
filew.close()