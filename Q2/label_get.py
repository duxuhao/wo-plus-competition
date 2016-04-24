# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 21:30:19 2016

@author: 21644336
"""

import pandas as pd
import numpy as np
from multiprocessing import Pool

pool = Pool(16)

df=pd.read_csv('model.csv',encoding='utf-8',header = None)

label = pd.Series(np.ones(len(df)))
n = 3
for i, model in enumerate(df[1][:]):
    try:
        a = str(model.lower().replace(" ","").replace("(","").replace(")",""))
        b = df[1][i+n-1:i+n].values[0].lower().replace(" ","").replace("(","").replace(")","")
        if bool(a.find(b)+1) | bool(b.find(a)+1):
            label[i]= 0
            print i
    except:
        label[i]= 0

name = 'Change_Phone_backward_' + str(n) +'.csv'
label.to_csv(name)