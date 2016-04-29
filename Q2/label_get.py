# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 21:30:19 2016

@author: 21644336
"""

import pandas as pd
import numpy as np
from multiprocessing import Pool

pool = Pool(16)

df=pd.read_csv('model.csv',encoding='utf-8')

label = pd.Series(np.ones(len(df)))
tt = df.Model
for i, model in enumerate(tt):
    try:
        a = str(model.lower().replace(" ","").replace("(","").replace(")",""))
        b = tt[i+2:i+3].values[0].lower().replace(" ","").replace("(","").replace(")","")
        if bool(a.find(b)+1) | bool(b.find(a)+1):
            label[i]= 0
            print i
    except:
        label[i]= 0

name = 'label2.csv'
df['label'] = pd.Series(label,index = df.index)
new = df[['IMSI','Month','label']]
new.to_csv(name,encoding = 'utf-8',index = None)