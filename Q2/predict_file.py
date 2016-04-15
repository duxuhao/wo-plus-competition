# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 21:44:26 2016

@author: 21644336
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier

df=pd.read_csv('Q2.csv',encoding='GBK')
label=pd.read_csv('label_3_months.csv',header=None)
label.columns = ['Index','label']
Brand=pd.read_csv('Price_Brand.csv')

#change the gender to number
df.loc[df.Gender == u'\u7537','Gender']=0
df.loc[df.Gender == u'\u5973','Gender']=1
df.loc[df.Gender == u'\u4e0d\u8be6','Gender']= 2

#change the Age to number
df.loc[df.Age == u'30-39','Age']=35
df.loc[df.Age == u'26-29','Age']=27.5
df.loc[df.Age == u'50-59','Age']= 55
df.loc[df.Age == u'40-49','Age']=45
df.loc[df.Age == u'18-22','Age']=20
df.loc[df.Age == u'23-25','Age']=24
df.loc[df.Age == u'60\u4ee5\u4e0a','Age']= 65
df.loc[df.Age == u'17\u5c81\u4ee5\u4e0b','Age']= 15
df.loc[df.Age == u'\u672a\u77e5','Age']= 0

#change the APRU to number
df.loc[df.APRU == u'50-99','APRU']=75
df.loc[df.APRU == u'0-49','APRU']=25
df.loc[df.APRU == u'300\u53ca\u4ee5\u4e0a','APRU']=350
df.loc[df.APRU == u'250-299','APRU']=275
df.loc[df.APRU == u'200-249','APRU']= 225
df.loc[df.APRU == u'150-199','APRU']=175
df.loc[df.APRU == u'100-149','APRU']= 125
df.loc[pd.isnull(df.APRU),'APRU'] = 0

#change the Network to number
df.loc[df.Network == u'3G','Network']=3
df.loc[df.Network == u'2G','Network']=2

#change the Network to number
df.loc[df.Flow == u'0-499','Flow']=250
df.loc[df.Flow == u'1000-1499','Flow']=1250
df.loc[df.Flow == u'1500-1999','Flow']=1750
df.loc[df.Flow == u'500-999','Flow']=750
df.loc[df.Flow == u'2500-2999','Flow']=2750
df.loc[df.Flow == u'3000-3499','Flow']=3250
df.loc[df.Flow == u'2000-2499','Flow']=2250
df.loc[df.Flow == u'3500-3999','Flow']=3750
df.loc[df.Flow == u'4000-4499','Flow']=4250
df.loc[df.Flow == u'5000\u4ee5\u4e0a','Flow']=5500
df.loc[df.Flow == u'4500-4999','Flow']=4750
df.loc[pd.isnull(df.Flow),'Flow'] = 0

new = pd.merge(df, Brand,on='Brand', left_index=True,how='left')
T =  ~pd.isnull(new.Price)
y=label.label[T.values][(df.Month != 201510) & (df.Month != 201511) & (df.Month != 201512)]
X = new[T][(new[T].Month != 201510) & (new[T].Month != 201511) & (new[T].Month != 201512)]
X_train, X_test, y_train, y_test=train_test_split(X[['Month','APRU','Flow','Call','SMS','Gender','Age','Network','Price']], y, test_size = 0.3)

clf =DecisionTreeClassifier()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)