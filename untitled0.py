# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 07:49:58 2020

@author: Administrator
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Customers.csv')

X = df.iloc[:,[3,4]].values

import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))

plt.title("Dendogram")
plt.xlabel('Customers')
plt.ylabel('Euclidean distance')
plt.show()



from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters= 3, affinity = 'euclidean',linkage = 'ward') 

y_hc = hc.fit_predict(X)



plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'cyan', label = '1st Cluster')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'green', label = '2nd Cluster')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'red', label = '3rd Cluster')
#plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'blue', label = '4th Cluster')

plt.title('Clusters of customers')
plt.xlabel('Annual Salary (k$)')
plt.ylabel('Spendings (1 to 100)')
plt.legend()
plt.show()





