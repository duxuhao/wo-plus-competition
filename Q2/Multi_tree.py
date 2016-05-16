import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier # this method give the best, haven't try ANN
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np

def predict(fea, df, t, t9, tr, clf, newname):
    Un = df.columns == 'Blank'
    for f in fea:
        Un = Un | (df.columns == f)
    y = df[t].label
    X = df[t].ix[:,Un]
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size = 0.9, random_state = 1)
    clf.fit(X_train, y_train)
    re = 'Testing AUC: \t' + str(roc_auc_score(y_test,clf.predict_proba(X_test)[:,1]))
    print re
    re =  'September AUC: \t' + str(roc_auc_score(df[t9].label,clf.predict_proba(df[t9].ix[:,Un])[:,1]))
    print re
    tr[str(newname)] = pd.Series(clf.predict_proba(df.ix[:,Un])[:,1], index = tr.index)
    #tr[str(newname)] = pd.Series(clf.predict(df.ix[:,Un]), index = tr.index)
    return Un, clf, tr

#a = pd.read_csv('Multi_tree.csv',encoding = 'utf-8')
'''
a = pd.read_csv('Q2Used.csv',encoding = 'utf-8')
label = pd.read_csv('change_phone_3_month.csv')
a['system'] = pd.Series(np.zeros(len(a)),index = a.index)
a.loc[a.Brand == 'Apple','system'] = 1
a = pd.merge(a, label, on =['IMSI','Month'], left_index = True, how = 'left')
a.loc[pd.isnull(a.Brand),'Brand'] = 'Unkown'
brands = a.Brand.value_counts()
brands_col = brands.index.tolist()
brands.index = range(len(brands))
df_brands = pd.DataFrame({"Brand":brands_col,"Brand_Counts":brands})
a = pd.merge(a,df_brands,how = "inner",on="Brand")
a = a[['IMSI','Month','label','compare2','compare1','last_change','Price','Trend_x','Trend_y','Result_Quantity','system','ave','Label','Call','SMS','Age','Network','Flow','APRU','Gender']]
'''
Fea = ['compare2','compare1','last_change','Price','Trend_x','Trend_y','Result_Quantity','system','ave','Label','Call','SMS','Age','Network','Flow','APRU','Gender']

T = (a.Month == 201501) |(a.Month == 201502) |(a.Month == 201503) |(a.Month == 201504) |(a.Month == 201505) |(a.Month == 201506)

T9 = (a.Month == 201509)

TreeResult = a[['IMSI','Month','label']]
clf_collect = [GradientBoostingClassifier(), AdaBoostClassifier(), RandomForestClassifier(), DecisionTreeClassifier(), ExtraTreesClassifier()]

for i, clf_used in enumerate(clf_collect[:2]):
    Un, clf, TreeResult= predict(fea = Fea, df = a, t = T, t9 = T9, tr = TreeResult, clf = clf_used, newname = i)

print '-'*40
print '-'*40
T = (TreeResult.Month == 201501) |(TreeResult.Month == 201502) |(TreeResult.Month == 201503) |(TreeResult.Month == 201504) |(TreeResult.Month == 201505) |(TreeResult.Month == 201506)
T9 = (TreeResult.Month == 201509)
clf = RandomForestClassifier()
y = TreeResult[T].label
X = TreeResult[T].ix[:,3:]
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size = 0.9, random_state = 1)
clf.fit(X_train, y_train)
re = 'Testing AUC: \t' + str(roc_auc_score(y_test,clf.predict_proba(X_test)[:,1]))
print re
re =  'September AUC: \t' + str(roc_auc_score(TreeResult[T9].label,clf.predict_proba(TreeResult[T9].ix[:,3:])[:,1]))
print re
