# -*- coding: utf-8 -*-
"""
Created on Sat Apr 09 13:53:39 2016

@author: 21644336
"""
# -*- coding: utf-8 -*-
import pandas as pd
Month = ["01","02","03","04","05","06","07","08","09","10","11","12"]
df = pd.DataFrame()
for month in Month:
    filename = "Q2" + "2015" + month + ".csv"
    test = pd.read_csv(filename,encoding="GBK")
    df =pd.concat([df, test], axis = 0)

df.columns = ['Month', 'IMSI','Network','Gender','Age','APRU','Brand','Model','Flow','Call','SMS']
df = df.sort_values(['IMSI','Month'])
df.Model = map(str.lower,df.Model)
df.Brand = map(str.lower,df.Brand)
