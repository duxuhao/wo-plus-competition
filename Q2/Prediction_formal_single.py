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
from sklearn.cross_validation import train_test_split
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

pool = Pool(8)
df=pd.read_csv('Q2_merge_all.csv',encoding='utf-8') # this used dataset with the basic privided parameter
test = pd.read_csv('Test_IMSI_all_used.csv',encoding = 'utf-8',header = None)
label=pd.read_csv('Change_Phone.csv',header=None) #the label for customer change cellphone or not
#The following is the clean and merge data process, the used one is saved as the Q2_merge_all.csv file
Test_Percentage = 0.3
test.columns = ['IMSI']
label.columns = ['Index','label'] #label the label

df= pd.merge(df, test, on='IMSI', right_index=True,how='inner')
label = label.iloc[df.index]

#zero round
T = (df.Month != 201510) & (df.Month != 201511) & (df.Month != 201512) # only the 12 months data provided, so label for 201510 to 201512
X = df[T]
#T = T.reset_index()
y=label.label[T]
#delete features that cannot be used
Unused_Feature = (df.columns != 'Month') & (X.columns != 'Brand') & (X.columns != 'Model') & (X.columns != 'IMSI')
train = X.ix[:,Unused_Feature]
X_train, X_test, y_train, y_test=train_test_split(train, y, test_size = Test_Percentage, random_state = np.random.randint(100000))
clf =RandomForestClassifier()
clf.fit(X_train, y_train)
auc = roc_auc_score(y_test,clf.predict_proba(X_test)[:,1])
print '0 Loop: ' + str(auc)

LoopNumber =7 #loop times
N = 3 #N months predict
for ite in range(LoopNumber):
    train = df.ix[:,Unused_Feature]
    prob = clf.predict_proba(train)
    T = (df.Month != 201510) & (df.Month != 201511) & (df.Month != 201512)
    for i in range(N):
        dfname1 = 'L' + str(ite * N + i + 1) + 'N'
        dfname2 = 'L' + str(ite * N + i + 1) + 'P'
        t1 = np.vstack((prob[-i-1:],prob[:-i-1]))
        #df[dfname1] = pd.Series(t1[:,0], index=df.index)
        df[dfname2] = pd.Series(t1[:,1], index=df.index)
        T = (df.Month != (201500 + i + 1)) & T  
    X = df[T]  
    #T = T.reset_index()
    y=label.label[T]
    Unused_Feature = (X.columns != 'Month') & (X.columns != 'Brand') & (X.columns != 'Model') & (X.columns != 'IMSI')
    train = X.ix[:,Unused_Feature]
    X_train, X_test, y_train, y_test=train_test_split(train, y, test_size = Test_Percentage, random_state = np.random.randint(100000))
    clf =RandomForestClassifier()    
    if (ite == LoopNumber-1):
           clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train)
    auc = roc_auc_score(y_test,clf.predict_proba(X_test)[:,1])
    print str(ite+1) + ' Loop: ' + str(auc)
    

sns.set_style('darkgrid')
plt.bar(range(len(X_test.columns)),clf.feature_importances_)
plt.xticks(np.arange(len(X_test.columns))+0.5,X_test.columns)
print 'Prediction Accuracy: ' + str(clf.score(X_test, y_test))
Unused_Feature = (X.columns != 'Month') & (X.columns != 'Brand') & (X.columns != 'Model') & (X.columns != 'IMSI')
train = X.ix[:,Unused_Feature]

train['score'] = pd.Series(predict_proba(train)[:,1], index=train.index)
pre = train[train.Month == 201512]
pre = pre[['IMSI','score']]
pre.columns = ['Idx','score']
pre.to_csv('Test_IMSI_all.csv',encoding = 'utf-8', index = None)
'''
Brand=pd.read_csv('Price_Brand_baidu.csv',encoding='utf-8') # the search result quantity of the coresponding brand
Model=pd.read_csv('Model_Price_u.csv') # the price of the model
Trend = pd.read_csv('Brand_Trend_Scale.csv',encoding = 'utf-8')
label.columns = ['Index','label'] #label the label
df = pd.merge(df, Brand,on='Brand', left_index=True,how='left')
df.loc[pd.isnull(df.Result_Quantity),'Result_Quantity'] = 0
df = pd.merge(df, Model,on='Model', left_index=True,how='left')
df.loc[pd.isnull(df.Price),'Price'] = 0

Test_Percentage = 0.3
label_range =np.array([0, 1e3, 1e4, 1e5, 1e6, 5e6, 1e7])
# the data for previous information
frame1 = []
frame2 = []
frame3 = []
frame4 = []
add = 'labelcompare1' #compare the the last month, phone change or not.
name = add + '2015.csv'
previous=pd.read_csv(name)
previous2=pd.read_csv('BrandSR.csv')
variable_name = add + '_ave'
for i in range(12):
    frame1.append(pd.DataFrame({'Month':np.ones(len(previous))*i + 201501,'IMSI':previous.IMSI, variable_name:np.mean(previous.ix[:,1:i+2],axis=1)}))
    frame2.append(pd.DataFrame({'Month':np.ones(len(previous))*i + 201501,'IMSI':previous.IMSI, 'previous_label':np.mean(previous.ix[:,i+1:i+2],axis=1)}))
    frame3.append(pd.DataFrame({'Month':np.ones(len(previous))*i + 201501,'IMSI':previous.IMSI, 'previous_label2':np.mean(previous.ix[:,i+0:i+1],axis=1)}))
    t = np.mean(previous2.ix[:,1:i+2],axis=1)
    tt = np.array([t,t,t,t,t,t,t]).T
    a = np.argmin(np.abs(tt - label_range),axis=1) +1
    frame4.append(pd.DataFrame({'Month':np.ones(len(previous2))*i + 201501,'IMSI':previous2.IMSI, 'Label':a}))

previous_change = pd.concat(frame1) 
previous_change.loc[pd.isnull(previous_change[variable_name]),variable_name] = 0
df = pd.merge(df, previous_change, on=['Month','IMSI'], left_index=True,how='left')
previous_label = pd.concat(frame2)
previous_label.loc[pd.isnull(previous_label['previous_label']),'previous_label'] = 0
df = pd.merge(df, previous_label, on=['Month','IMSI'], left_index=True,how='left')
previous_label2 = pd.concat(frame3) 
previous_label2.loc[pd.isnull(previous_label2['previous_label2']),'previous_label2'] = 0
df = pd.merge(df, previous_label2, on=['Month','IMSI'], left_index=True,how='left')
previous_brand = pd.concat(frame4)
previous_brand.loc[pd.isnull(previous_brand['Label']),'Label'] = 1
df = pd.merge(df, previous_brand, on=['Month','IMSI'], left_index=True,how='left')
df = pd.merge(df, Trend,on=['Month','Label'], left_index=True,how='left')
'''

'''
#second round
#third round
train = df.ix[:,(df.columns != 'Month') & (X.columns != 'Brand') &(df.columns != 'Model') & (df.columns != 'IMSI') & (df.columns != 'Unnamed: 0')]
prob = clf.predict_proba(train)
t1 = np.vstack((prob[-2:],prob[:-2]))
df['L2P'] = pd.Series(t1[:,0], index=df.index)
df['L2N'] = pd.Series(t1[:,1], index=df.index)

T =(df.Month != 201510) & (df.Month != 201511) & (df.Month != 201512) & (df.Month != 201501) & (df.Month != 201502)
T = T.reset_index()
y=label.label[T.Month]
X = df[(df.Month != 201510) & (df.Month != 201511) & (df.Month != 201512)& (df.Month != 201501) & (df.Month != 201502)]
Unused_Feature = (X.columns != 'Brand') & (X.columns != 'Model') & (X.columns != 'IMSI') & (X.columns != 'Unnamed: 0')
train = X.ix[:,(df.columns != 'Month') & (X.columns != 'Brand') & (X.columns != 'Model') & (X.columns != 'IMSI') & (X.columns != 'Unnamed: 0')]
X_train, X_test, y_train, y_test=train_test_split(train, y, test_size = Test_Percentage, random_state = np.random.randint(100000))

clf =RandomForestClassifier()
clf.fit(X_train, y_train)
auc = roc_auc_score(y_test,clf.predict_proba(X_test)[:,1])
print 'Third Loop: ' + str(auc)

#third round
train = df.ix[:,(df.columns != 'Month') & (X.columns != 'Brand') &(df.columns != 'Model') & (df.columns != 'IMSI') & (df.columns != 'Unnamed: 0')]
prob = clf.predict_proba(train)
t1 = np.vstack((prob[-3],prob[:-3]))
df['L3P'] = pd.Series(t1[:,0], index=df.index)
df['L3N'] = pd.Series(t1[:,1], index=df.index)

T =(df.Month != 201503) & (df.Month != 201510) & (df.Month != 201511) & (df.Month != 201512) & (df.Month != 201501) & (df.Month != 201502)
T = T.reset_index()
y=label.label[T.Month]
X = df[(df.Month != 201503) & (df.Month != 201510) & (df.Month != 201511) & (df.Month != 201512) & (df.Month != 201501) & (df.Month != 201502)]
Unused_Feature = (X.columns != 'Brand') & (X.columns != 'Model') & (X.columns != 'IMSI') & (X.columns != 'Unnamed: 0')
train = X.ix[:,(df.columns != 'Month') & (X.columns != 'Brand') & (X.columns != 'Model') & (X.columns != 'IMSI') & (X.columns != 'Unnamed: 0')]
X_train, X_test, y_train, y_test=train_test_split(train, y, test_size = Test_Percentage, random_state = np.random.randint(100000))

clf =RandomForestClassifier()
clf.fit(X_train, y_train)
auc = roc_auc_score(y_test,clf.predict_proba(X_test)[:,1])
print 'Fourth Loop: ' + str(auc)

#fourth round
train = df.ix[:,(df.columns != 'Month') & (X.columns != 'Brand') &(df.columns != 'Model') & (df.columns != 'IMSI') & (df.columns != 'Unnamed: 0')]
prob = clf.predict_proba(train)
t1 = np.vstack((prob[-1],prob[:-1]))
df['L4P'] = pd.Series(t1[:,0], index=df.index)
df['L4N'] = pd.Series(t1[:,1], index=df.index)

T =(df.Month != 201503) & (df.Month != 201504) & (df.Month != 201510) & (df.Month != 201511) & (df.Month != 201512) & (df.Month != 201501) & (df.Month != 201502)
T = T.reset_index()
y=label.label[T.Month]
X = df[(df.Month != 201503) & (df.Month != 201504) & (df.Month != 201510) & (df.Month != 201511) & (df.Month != 201512) & (df.Month != 201501) & (df.Month != 201502)]
Unused_Feature = (X.columns != 'Brand') & (X.columns != 'Model') & (X.columns != 'IMSI') & (X.columns != 'Unnamed: 0')
train = X.ix[:,(df.columns != 'Month') & (X.columns != 'Brand') & (X.columns != 'Model') & (X.columns != 'IMSI') & (X.columns != 'Unnamed: 0')]
X_train, X_test, y_train, y_test=train_test_split(train, y, test_size = Test_Percentage, random_state = np.random.randint(100000))

clf =GradientBoostingClassifier()
clf.fit(X_train, y_train)
auc = roc_auc_score(y_test,clf.predict_proba(X_test)[:,1])
print 'Fifth Loop: ' + str(auc)

#fifth round
train = df.ix[:,(df.columns != 'Month') & (X.columns != 'Brand') &(df.columns != 'Model') & (df.columns != 'IMSI') & (df.columns != 'Unnamed: 0')]
prob = clf.predict_proba(train)
t1 = np.vstack((prob[-1],prob[:-1]))
df['L5P'] = pd.Series(t1[:,0], index=df.index)
df['L5N'] = pd.Series(t1[:,1], index=df.index)

T =(df.Month != 201503) & (df.Month != 201504) & (df.Month != 201505) & (df.Month != 201510) & (df.Month != 201511) & (df.Month != 201512) & (df.Month != 201501) & (df.Month != 201502)
T = T.reset_index()
y=label.label[T.Month]
X = df[(df.Month != 201503) & (df.Month != 201504) & (df.Month != 201505) & (df.Month != 201510) & (df.Month != 201511) & (df.Month != 201512) & (df.Month != 201501) & (df.Month != 201502)]
Unused_Feature = (X.columns != 'Brand') & (X.columns != 'Model') & (X.columns != 'IMSI') & (X.columns != 'Unnamed: 0')
train = X.ix[:,(df.columns != 'Month') & (X.columns != 'Brand') & (X.columns != 'Model') & (X.columns != 'IMSI') & (X.columns != 'Unnamed: 0')]
X_train, X_test, y_train, y_test=train_test_split(train, y, test_size = Test_Percentage, random_state = np.random.randint(100000))

clf =GradientBoostingClassifier()
clf.fit(X_train, y_train)
auc = roc_auc_score(y_test,clf.predict_proba(X_test)[:,1])
print 'Sixth Loop: ' + str(auc)
'''
'''

'''