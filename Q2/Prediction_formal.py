# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 20:34:10 2016

@author: 21644336
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier , RandomForestClassifier # this method give the best, haven't try ANN
from sklearn.cross_validation import train_test_split
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

pool = Pool(8)
df=pd.read_csv('Q2_clean.csv',encoding='utf-8') # this used dataset with the basic privided parameter
label=pd.read_csv('Change_Phone.csv',header=None) #the label for customer change cellphone or not
Brand=pd.read_csv('Brand_baidu.csv',encoding='utf-8') # the search result quantity of the coresponding brand
Model=pd.read_csv('Price_Model.csv') # the price of the model
Trend = pd.read_csv('Brand_Trend_Scale.csv',encoding = 'utf-8')
label.columns = ['Index','label'] #label the label


# the data for previous information
frame1 = []
frame2 = []
frame3 = []
Test_Percentage = 0.3
add = 'labelcompare1' #this one must be added as it take 30% importance
name = add + '2015.csv'
previous=pd.read_csv(name)
variable_name = add + '_ave'
for i in range(12):
    frame1.append(pd.DataFrame({'Month':np.ones(len(previous))*i + 201501,'IMSI':previous.IMSI, variable_name:np.mean(previous.ix[:,1:i+2],axis=1)}))
    frame2.append(pd.DataFrame({'Month':np.ones(len(previous))*i + 201501,'IMSI':previous.IMSI, 'previous_label':np.mean(previous.ix[:,i+1:i+2],axis=1)}))
    frame3.append(pd.DataFrame({'Month':np.ones(len(previous))*i + 201501,'IMSI':previous.IMSI, 'previous_label2':np.mean(previous.ix[:,i+0:i+1],axis=1)}))

previous_change = pd.concat(frame1) 
previous_change.loc[pd.isnull(previous_change[variable_name]),variable_name] = 0
df = pd.merge(df, previous_change, on=['Month','IMSI'], left_index=True,how='left')
previous_label = pd.concat(frame2)
previous_label.loc[pd.isnull(previous_label['previous_label']),'previous_label'] = 0
df = pd.merge(df, previous_label, on=['Month','IMSI'], left_index=True,how='left')
previous_label2 = pd.concat(frame3) 
previous_label2.loc[pd.isnull(previous_label2['previous_label2']),'previous_label2'] = 0
df = pd.merge(df, previous_label2, on=['Month','IMSI'], left_index=True,how='left')

new = pd.merge(df, Brand,on='Brand', left_index=True,how='left')
new.loc[pd.isnull(new.Result_Quantity),'Result_Quantity'] = 0
new.loc[pd.isnull(new.Label),'Label'] = 1
new = pd.merge(new, Model,on='Model', right_index=True,how='left')
new = pd.merge(new, Trend,on=['Month','Label'], left_index=True,how='left')

new.loc[pd.isnull(new.Price),'Price'] = 0

T = (df.Month != 201510) & (df.Month != 201511) & (df.Month != 201512) # only the 12 months data provided, so label for 201510 to 201512
T = T.reset_index()
y=label.label[T.Month]
X = new[(new.Month != 201510) & (new.Month != 201511) & (new.Month != 201512)]

#delete features that cannot be used
Unused_Feature = (X.columns != 'Month') & (X.columns != 'Brand') & (X.columns != 'Model') & (X.columns != 'IMSI') & (X.columns != 'Unnamed: 0')
train = X.ix[:,Unused_Feature]
X_train, X_test, y_train, y_test=train_test_split(train, y, test_size = Test_Percentage, random_state = np.random.randint(100000))

clf =RandomForestClassifier()
clf.fit(X_train, y_train)

auc = roc_auc_score(y_test,clf.predict_proba(X_test)[:,1])
print(auc)

#second round
train = new.ix[:,Unused_Feature]
prob = clf.predict_proba(train)
t1 = np.vstack((prob[-1],prob[:-1]))
t2 = np.vstack((prob[-2],prob[-1],prob[:-2]))
new['L1P'] = pd.Series(t1[:,0], index=new.index)
new['L1N'] = pd.Series(t1[:,1], index=new.index)
new['L2P'] = pd.Series(t2[:,0], index=new.index)
new['L2N'] = pd.Series(t2[:,1], index=new.index)

T = (df.Month != 201510) & (df.Month != 201511) & (df.Month != 201512) & (df.Month != 201501) & (df.Month != 201502)
T = T.reset_index()
y=label.label[T.Month]
X = new[(new.Month != 201510) & (new.Month != 201511) & (new.Month != 201512)& (new.Month != 201501) & (new.Month != 201502)]
Unused_Feature = (X.columns != 'Month') & (X.columns != 'Brand') & (X.columns != 'Model') & (X.columns != 'IMSI') & (X.columns != 'Unnamed: 0')
train = X.ix[:,Unused_Feature]
X_train, X_test, y_train, y_test=train_test_split(train, y, test_size = Test_Percentage, random_state = np.random.randint(100000))

clf =GradientBoostingClassifier()
clf.fit(X_train, y_train)
auc = roc_auc_score(y_test,clf.predict_proba(X_test)[:,1])
print(auc)
sns.set_style('darkgrid')
plt.bar(range(len(X_test.columns)),clf.feature_importances_)
plt.xticks(np.arange(len(X_test.columns))+0.5,X_test.columns)
