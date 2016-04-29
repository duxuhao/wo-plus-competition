# -*- coding: utf-8 -*-
"""
Created on Sat Apr 09 13:53:39 2016

@author: 21644336
"""
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
#this code is commented if you got the output file
'''
Month = ["01","02","03","04","05","06","07","08","09","10","11","12"]
df = pd.DataFrame()
for month in Month:
    filename = "Q2" + "2015" + month + ".csv"
    test = pd.read_csv(filename,encoding="GBK")
    df =pd.concat([df, test], axis = 0)

df.columns = ['Month', 'IMSI','Network','Gender','Age','APRU','Brand','Model','Data','Call','SMS']
df = df.sort_values(['IMSI','Month'])
'''
#range to digit
df=pd.read_csv('Q2.csv',encoding='GBK')
Brand=pd.read_csv('Price_Brand_baidu.csv',encoding='utf-8') # the search result quantity of the coresponding brand
Model=pd.read_csv('Model_Price_u.csv') # the price of the model
Trend = pd.read_csv('Brand_Trend_Scale.csv',encoding = 'utf-8')
previous2=pd.read_csv('BrandSR.csv')
Dic=pd.read_csv('Dic.csv',encoding = 'utf-8')


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

df.to_csv('Temp.csv',index=None,encoding = 'utf-8')
df = pd.read_csv('Temp.csv',encoding = 'utf-8')

#brand chinese to english
Dic.columns = ['Brand', 'Chi']
for i in range(len(Dic)):
    a = df.Brand.replace(Dic.Chi[i],Dic.Brand[i],inplace = True)

df = pd.merge(df, Brand,on='Brand', left_index=True,how='left')
df.loc[pd.isnull(df.Result_Quantity),'Result_Quantity'] = 0
df = pd.merge(df, Model,on='Model', left_index=True,how='left')
df.loc[pd.isnull(df.Price),'Price'] = 0

label_range =np.array([0, 1e3, 1e4, 1e5, 1e6, 5e6, 1e7])
# the data for previous information
frame1 = []
frame2 = []
frame3 = []
frame4 = []
add = 'labelcompare1' #compare the the last month, phone change or not.
name = add + '2015.csv'
previous=pd.read_csv(name)
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

df.to_csv('Q2_used.csv',encoding = 'utf-8',index = None)