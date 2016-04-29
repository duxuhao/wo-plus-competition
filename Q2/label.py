# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 10:06:23 2016

@author: 21644336
"""

import pandas as pd
import numpy as np
from multiprocessing import Pool

pool = Pool(8)
df = pd.read_csv('Q2_merge_all.csv',encoding = 'utf-8')
df['Change'] = pd.Series(np.zeros(len(df)), index = df.index)
n = 0
for imsi in df.IMSI.unique():
    for i in range(9):
        a = df[df.Month == 201501 + i][df.IMSI == imsi].Model.values[0].replace(' ','').replace(')','').replace('(','').lower()
        b = df[df.Month == 201504 + i][df.IMSI == imsi].Model.values[0].replace(' ','').replace(')','').replace('(','').lower()
        t = bool(a.find(b) + 1) | bool(b.find(a) + 1)
        df.loc[(df.Month == 201501 + i) & (df.IMSI == imsi),'Change'] = int(t)
        n += 1 
        print(n)

Label = df[['IMSI','Month','label']]
Label.to_csv('label.csv',encoding = 'utf-8',index = None)