# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 14:31:40 2016

read: Q22015+month(=1,2,3,...12).csv
    csv files containing the mobile usage of 500,000 usrs. each for a month
write:
    apru.csv, brand.csv, call.csv, flow.csv, model.csv, sms.csv
    csv files containing corresponding information of a time-tracking record
    of each individual predictor. 

@author: 21355188
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import pandas as pd
import numpy as np



Month = ["01","02","03","04","05","06","07","08","09","10","11","12"]
df = pd.DataFrame()

for month in Month:
    filename = "Q2" + "2015" + month + ".csv"
    temp = pd.read_csv(filename,encoding="GBK")
    temp.columns = ['Month', 'IMSI','Network','Gender','Age','APRU','Brand','Model','Flow','Call','SMS']  
    temp = temp.sort_values("IMSI")
    temp = temp.set_index([range(500000)])
    
    #change the gender to number
    temp.loc[temp.Gender == u'\u7537','Gender']=0
    temp.loc[temp.Gender == u'\u5973','Gender']=1
    temp.loc[temp.Gender == u'\u4e0d\u8be6','Gender']= 2

    #change the Age to number
    temp.loc[temp.Age == u'30-39','Age']=35
    temp.loc[temp.Age == u'26-29','Age']=27.5
    temp.loc[temp.Age == u'50-59','Age']= 55
    temp.loc[temp.Age == u'40-49','Age']=45
    temp.loc[temp.Age == u'18-22','Age']=20
    temp.loc[temp.Age == u'23-25','Age']=24
    temp.loc[temp.Age == u'60\u4ee5\u4e0a','Age']= 65
    temp.loc[temp.Age == u'17\u5c81\u4ee5\u4e0b','Age']= 15
    temp.loc[temp.Age == u'\u672a\u77e5','Age']= 0

    #change the APRU to number
    temp.loc[temp.APRU == u'50-99','APRU']=75
    temp.loc[temp.APRU == u'0-49','APRU']=25
    temp.loc[temp.APRU == u'300\u53ca\u4ee5\u4e0a','APRU']=350
    temp.loc[temp.APRU == u'250-299','APRU']=275
    temp.loc[temp.APRU == u'200-249','APRU']= 225
    temp.loc[temp.APRU == u'150-199','APRU']=175
    temp.loc[temp.APRU == u'100-149','APRU']= 125
    temp.loc[temp["APRU"].isnull(),'APRU'] = 0
    
    #change the Network to number
    temp.loc[temp.Network == u'3G','Network']=3
    temp.loc[temp.Network == u'2G','Network']=2

    #change the Network to number
    temp.loc[temp.Flow == u'0-499','Flow']=250
    temp.loc[temp.Flow == u'1000-1499','Flow']=1250
    temp.loc[temp.Flow == u'1500-1999','Flow']=1750
    temp.loc[temp.Flow == u'500-999','Flow']=750
    temp.loc[temp.Flow == u'2500-2999','Flow']=2750
    temp.loc[temp.Flow == u'3000-3499','Flow']=3250
    temp.loc[temp.Flow == u'2000-2499','Flow']=2250
    temp.loc[temp.Flow == u'3500-3999','Flow']=3750
    temp.loc[temp.Flow == u'4000-4499','Flow']=4250
    temp.loc[temp.Flow == u'5000\u4ee5\u4e0a','Flow']=5500
    temp.loc[temp.Flow == u'4500-4999','Flow']=4750
    temp.loc[temp["Flow"].isnull(),'Flow'] = 0
    
        
    df =pd.concat([df, temp], axis = 1)
    imsi = temp.loc[:,'IMSI']
    gender = temp.loc[:,'Gender']
    age = temp.loc[:,'Age']
    
    
'''
Bind each predictor with IMSI 
'''
age = pd.concat([imsi,age],axis = 1)
gender = pd.concat([imsi,gender],axis =1)
network = df.loc[:,"Network"]
network = pd.concat([imsi,network],axis =1)
apru = df.loc[:,"APRU"]
apru = pd.concat([imsi,apru],axis =1)
brand = df.loc[:,"Brand"]
brand = pd.concat([imsi,brand],axis =1)
model = df.loc[:,"Model"]
model = pd.concat([imsi,model],axis =1)
flow = df.loc[:,"Flow"]
flow = pd.concat([imsi,flow],axis =1)
call = df.loc[:,"Call"]
call = pd.concat([imsi,call],axis =1)
sms = df.loc[:,"SMS"]
sms = pd.concat([imsi,sms],axis =1)




"""
Export each predictor csv (of 12 months)
sample:
network.csv
index network  network  ....  network  
0      ###      ###             ###
1      ###      ###             ###
2      ###      ###             ###
.
.
.
"""
network.to_csv("network2015.csv",index = False)
apru.to_csv("apru2015.csv",index = False)
brand.to_csv("brand2015.csv",index = False,encoding = "utf-8")
model.to_csv("model2015.csv",index = False,encoding = "utf-8")
flow.to_csv("flow2015.csv",index = False)
call.to_csv("call2015.csv",index = False)
sms.to_csv("sms2015.csv",index = False)
