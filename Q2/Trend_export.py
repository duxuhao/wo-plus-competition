# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:35:30 2016

@author: 21644336
"""

import pandas as pd
import numpy as np

Brand = pd.read_csv('Brand_baidu.csv',encoding = 'utf-8')
Index = pd.read_csv('Brand_serach_index_360_201603.csv',encoding = 'utf-8')
new = pd.merge(Index, Brand[['Brand','Label']], on = 'Brand', left_index=True, how='left')
#month = ['Brand','201501','201502','201503','201504','201505','201506','201507','201508','201509','201510','201511','201512','201601','201602','201603','Label']
month = ['Brand','201412','201501','201502','201503','201504','201505','201506','201507','201508','201509','201510','201511','201512','201601','201602','Label']
new.columns = [month]
frame1 = []
n = 0
for l in range(1,8):
    for m in month[1:-1]:
        T = (new.Label == l)
        a = sum(new[T][m])
        frame1.append(pd.DataFrame({'Month':m, 'Label':l, 'Trend':a}, index=[n]))
        n += 1
      
df = pd.concat(frame1)
for i in range(3,8):
    df.loc[df.Label == i, 'Trend'] = df[df.Label == i]['Trend']/max(df[df.Label == i]['Trend'])

df.to_csv('Trend_new.csv',encoding = 'utf-8', index = None)