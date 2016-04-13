# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np

df = pd.read_csv("wo.csv",encoding = "gb2312")
#df[1:3]
df.rename(columns={u'月银行短信通知次数': 'BankMessages'}, inplace=True)
df.rename(columns={u'使用汽车类APP的PV数': 'AutoApp'}, inplace=True)
df.rename(columns={u'使用理财类APP的PV数': 'FinanApp'}, inplace=True)
df.rename(columns={u'使用股票类APP的PV数': 'StockApp'}, inplace=True)
df.rename(columns={u'交往圈规模': 'SocialNetworkSize'}, inplace=True)
df.rename(columns={u'是否有跨省行为': 'CrossProvinceExp'}, inplace=True)
df.rename(columns={u'是否有出国行为': 'OverseaExp'}, inplace=True)
df.rename(columns={u'访问购物类网站的次数': 'browseShoppingSites'}, inplace=True)
df.rename(columns={u'访问IT类网站的次数': 'browseITSites'}, inplace=True)
df.rename(columns={u'访问餐饮类网站的次数': 'browseRestaurantSites'}, inplace=True)
df.rename(columns={u'访问房产类网站的次数': 'browseRealEstateSites'}, inplace=True)
df.rename(columns={u'访问健康类网站的次数': 'browseHealthSites'}, inplace=True)
df.rename(columns={u'访问金融类网站的次数': 'browseFinanceSites'}, inplace=True)
df.rename(columns={u'访问旅游类网站的次数': 'browseTravelSites'}, inplace=True)
df.rename(columns={u'访问体育类网站的次数': 'browseSportsSites'}, inplace=True)
df.rename(columns={u'访问汽车类网站的次数': 'browseAutoSites'}, inplace=True)
df.rename(columns={u'访问时事类网站的次数': 'browseNewsSites'}, inplace=True)
df.rename(columns={u'访问社会类网站的次数': 'browseCommunitySites'}, inplace=True)
df.rename(columns={u'访问文娱类网站的次数': 'browseRecreationSites'}, inplace=True)
df.rename(columns={u'访问招聘类网站的次数': 'browseJobsSites'}, inplace=True)
df.rename(columns={u'访问教育类网站的次数': 'browseEducationSites'}, inplace=True)
df.rename(columns={u'访问其他类网站的次数': 'browseOtherSites'}, inplace=True)
df.rename(columns={u'访问网游类网站的次数': 'browseOnlineGamingSites'}, inplace=True)

#df["CrossProvinceExp"][df["CrossProvinceExp"] == u'\u662f'] = 1 # replace Yes with 1
df.loc[df["CrossProvinceExp"] == u'\u662f','CrossProvinceExp'] = 1 # replace Yes with 1
df["CrossProvinceExp"][df["CrossProvinceExp"] == u'\u5426'] = 0 # replace No with 0

df["OverseaExp"][df["OverseaExp"] == u'\u662f'] = 1 # replace Yes with 1
df["OverseaExp"][df["OverseaExp"] == u'\u5426'] = 0 # replace No with 0




mask1 = (df["CrossProvinceExp"] != 0) & (df["CrossProvinceExp"] != 1)
check1 = len(df["CrossProvinceExp"][mask1])

mask2 = (df["OverseaExp"] != 0) & (df["OverseaExp"] != 1)
check2 = len(df["OverseaExp"][mask2])

df.to_csv('wo2.csv', index=False)
print('Done')
#df1 = df[1:10]