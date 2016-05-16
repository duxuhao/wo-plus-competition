# -*- coding: utf-8 -*-
"""
Created on Wed May 04 00:27:38 2016

@author: 21644336
"""

a = df.copy()
T1_6 = (df.Month == 201501) |(df.Month == 201502) |(df.Month == 201503) |(df.Month == 201504) |(df.Month == 201505) |(df.Month == 201506) |(df.Month == 201507) |(df.Month == 201508) |(df.Month == 201509)
T9 = (df.Month == 201509)
y = a[T1_6].label
Fea = ['Network','Gender','Age','APRU','Flow','Call','SMS','Result_Quantity','Price','Label','Trend_x','ave','system','Brand_Counts','last_change','compare2'] 
Un = a.columns == 'B'
for f in Fea:
	Un = Un | (a.columns == f)
 
test = pd.read_csv('Test_IMSI_all_used.csv',encoding = 'utf-8')

X = a[T1_6].ix[:,Un]
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size = 0.9, random_state = 1)
clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)
roc_auc_score(y_test,clf.predict_proba(X_test)[:,1])

a['score'] = pd.Series(clf.predict_proba(a.ix[:,Un])[:,1], index = a.index)
pre = a[a.Month == 201512][['IMSI','score']]
pre.columns = ['Idx','score']
test.columns = ['Idx']

new = pd.merge(test, pre,on='Idx', left_index=True,how='left')
name = 'woplus_submit_sample.csv'
new.to_csv(name,encoding = 'utf-8', index = None)

a = df[['IMSI','Month','APRU','Network','Flow','Call','SMS','Result_Quantity','Price','Brand_Counts','system','Trend_x','last_change','Label']]
a.Month += 1
new = pd.merge(df, a, on =['IMSI','Month'], left_index = True, how = 'left')