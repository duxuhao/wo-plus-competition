# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 00:21:58 2016

@author: 21644336
"""
import pandas as pd
#import numpy as np
import requests
#from bs4 import BeautifulSoup

#T = np.load("Brand.npy")
#df = pd.read_csv("brandmodel.csv",encoding='GBK')
T = pd.read_csv("Brand_Model_unique.csv",encoding='utf-8')
#T = df[["Brand","Model"]]
#T = T.drop_duplicates()
Price_file = open('Price_Brand_Model.csv','a')

Price_file.write('Model')
Price_file.write(',')
Price_file.write('Price')

n=0
for i in range(700,710):
    Price_file.write('\n')
    try:
        Price_file.write(str(T.Model[i]))
    
        try:
            name = T.Brand[i].replace(" ","+") + "+" + T.Model[i].replace(" ","+")
            website = "http://search.zol.com.cn/s/all.php?kword=" + name
            
            req = requests.get(website)
            a = req.text
            money = a.find(u'\uffe5')
            
            if money == -1:
                money = a.find(u'\xa5')
            if money == -1:
                website = "http://www.baidu.com/s?wd=" + name
                req = requests.get(website)
                a = req.text
                money = a.find(u'\uffe5')
                if money == -1:
                    money = a.find(u'\xa5')
            #print website
            price = a[money+1:money+5].replace(" ","").replace("<","").replace("-","").replace(".","").replace("]","").replace(")","").replace("[","").replace("(","")
            #soup = BeautifulSoup(a,'html.parser')
            #price = soup.em.string[1:]
            Price_file.write(',')
            #price = len(str(soup))
            Price_file.write(str(price))
            print str(T.Model[i]) + "      " + str(price)
        except:
            n += 0.01
            Price_file.write(',')
            Price_file.write(str(n))
            pass
    except:
            pass
    
    
Price_file.close()