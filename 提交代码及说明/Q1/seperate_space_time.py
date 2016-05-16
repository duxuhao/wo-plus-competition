# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 21:03:16 2016

@author: 21644336
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('shujudasai_1.csv',header = None)
df.columns = ['date','IMEI','A00','A01','A10','A11','A20','A21','A30','A31','A40','A41','A50','A51','A60','A61','A70','A71','A80','A81','A90','A91','A100','A101','A110','A111','P00','P01','P10','P11','P20','P21','P30','P31','P40','P41','P50','P51','P60','P61','P70','P71','P80','P81','P90','P91','P100','P101','P110','P111']
#IMEI = df.IMEI.unique()
#Group = pd.read_csv('shujudasai_1.csv',header = None)
for i in range(len(Group)):
    Group_space_time = df[df.IMEI == Group[i]]
    
Group_space_time.to_csv('Group1.csv')