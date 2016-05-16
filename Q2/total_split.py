# -*- coding: utf-8 -*-
"""
Created on Sat May 07 21:35:21 2016

@author: 21644336
"""

def predict(fea, df, t, t9):
    Un = df.columns == 'Blank'
    for f in fea:
        '''        
        try:
            df[(f+'_y')] = df[(f+'_x')] - df[(f+'_y')]
            print(1)
        except:
            pass
        '''
        Un = Un | (df.columns == f)
        Un = Un | (df.columns == (f+'_x'))
        Un = Un | (df.columns == (f+'_y'))
    
    clf = GradientBoostingClassifier()
    y = df[t].label
    X = df[t].ix[:,Un]
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size = 0.9, random_state = 1)
    '''
    y = df[t & (df.label==0)].label
    X = df[t & (df.label==0)].ix[:,Un]
    X_train0, X_test0, y_train0, y_test0=train_test_split(X, y, test_size = 0.94, random_state = 1)
    y = df[t & (df.label==1)].label
    X = df[t & (df.label==1)].ix[:,Un]
    X_train1, X_test1, y_train1, y_test1=train_test_split(X, y, test_size = 0.8, random_state = 1)    
    X_train = pd.concat([X_train0, X_train1])
    y_train = pd.concat([y_train0, y_train1])
    X_test = pd.concat([X_test0, X_test1])
    y_test = pd.concat([y_test0, y_test1])
    '''
    clf.fit(X_train, y_train)
    print(len(y_train))
    print(sum(y_train))
    re = 'Testing AUC: \t' + str(roc_auc_score(y_test,clf.predict_proba(X_test)[:,1]))
    print re
    re =  'September AUC: \t' + str(roc_auc_score(df[t9].label,clf.predict_proba(df[t9].ix[:,Un])[:,1]))
    print re
    return Un, clf
'''
Fea = ['Network_x','Gender_x','Call_x','SMS_x','Result_Quantity_x','Price_x','Label_x','Brand_Counts_x',
       'last_change_x','compare1_x','monthly_brand_counts_x','monthly_brand_counts_y','Result_Quantity_y',
       'compare1_y','ave_y','APRU_y','Gender_y','Age_y','SMS_y','compare2_y','last_change_y','Call_y']
'''
#Fea =['Label', 'Price', 'Result_Quantity', 'compare1', 'Brand_Counts', 'ave', 'APRU', 'Gender', 'Network', 'Age', 'Flow', 'SMS', 'compare2', 'last_change', 'Call']
Fea = ['Brand_Counts','last_change','Trend_x','Result_Quantity','system', 'compare1','Price','Label']# total > 1 0.596 
#Fea =['Label', 'Price', 'Result_Quantity', 'compare1', 'ave', 'APRU', 'Gender', 'Network', 'Age', 'SMS', 'compare2', 'last_change', 'Call'] # total < 2 0.632 test_size 0.8 (1), 0.94(0)

x = (a.Total  > 1)
T = (a.Month == 201502) |(a.Month == 201502) |(a.Month == 201503) |(a.Month == 201504) |(a.Month == 201505) |(a.Month == 201506) #|(a.Month == 201507) |(a.Month == 201508) |(a.Month == 201509)
T9 = (a.Month == 201509) & x
Un, clf = predict(fea = Fea, df = a, t = T, t9 = T9 & x)