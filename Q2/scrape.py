# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 00:21:58 2016

@author: 21644336
"""

import numpy as np
import requests
from bs4 import BeautifulSoup

T = np.load("Phone_Name.npy")
Price_file = open('Price.csv','a')
'''
Price_file.write('Model')
Price_file.write(',')
Price_file.write('Price')
'''
n=0.3
for i in range(469,len(T)):
    Price_file.write('\n')
    try:
        Price_file.write(str(T[i]))
    
        try:
            website = "http://search.zol.com.cn/s/all.php?kword=" + T[i].replace(" ","+")
            req = requests.get(website)
            a = req.text
            soup = BeautifulSoup(a,'html.parser')
            price = soup.em.string[1:]
            Price_file.write(',')
            Price_file.write(price)
            print str(T[i]) + "      " + price
        except:
            n += 0.01
            Price_file.write(',')
            Price_file.write(str(n))
            pass
    except:
            pass
    
    
Price_file.close()