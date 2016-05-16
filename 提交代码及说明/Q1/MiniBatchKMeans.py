# -*- coding: utf-8 -*-
"""
===================================

Finds core samples of high density and expands clusters from them.

"""
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans


##############################################################################
# read the data 
os.chdir("D:\Data_competition\wo_plus")
data1 = pd.read_csv("wo2.csv")
X= data1.drop(['IMEI'], axis=1)
# remove the NAN and standlize
X=X.fillna(0)
X= StandardScaler().fit_transform(X)

##############################################################################
# Compute MiniBatchKMeans

mbk = MiniBatchKMeans(init='k-means++', n_clusters=15, batch_size=100000,
                      n_init=10, max_no_improvement=30, verbose=0,
                      random_state=2016)
mbk.fit(X)
mbk_means_labels_unique = np.unique(mbk.labels_)

id=data1['IMEI'].to_frame()
id['label']=mbk.labels_
id.to_csv("labled.csv")

##############################################################################
# inload the labeled results 
Labeled=pd.read_csv("labled.csv")
GeoInform=pd.read_csv("shujudasai_1.csv",header=None)

## merge two documents and export 
LabeledGeo=pd.merge(Labeled,GeoInform,left_index=1,right_index=1)
LabeledGeo.sort(columns="IMEI")
LabeledGeo.to_csv("LabeledGeo.csv")
