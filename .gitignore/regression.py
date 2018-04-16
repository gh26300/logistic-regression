# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 13:07:59 2018

@author: User
"""

#from numpy import * 
#from operator import * 
#import pandas as pd
#import numpy as np
#from pandas import Series, DataFrame
#
#
#!/usr/bin/env python

# encoding: utf-8
from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

'函式  https://github.com/hanxlinsist/jupyter_hub/blob/master/csdn/tools.py'
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.01):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],alpha=0.8, c=cmap(idx),marker=markers[idx], label=cl)
    # highlight test samples
    if test_idx:
        X_test = X[test_idx, :]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidth=1, marker='o', s=100, label='test set')


'sklearn 內建 iris data'
iris = datasets.load_iris()
'用 2 3列做為regression data'
x=iris.data[:,[1,2]]
y=iris.target
'for test  3D visual data'
X_reduced = PCA(n_components=3).fit_transform(iris.data)
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,cmap=plt.cm.Paired)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])
plt.show()


'用sklearn tarin test 分離測試與訓練資料'
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

sc=StandardScaler()
'計算data Mean & Standard Deviation'
sc.fit(x_train)
'取平均'
#sc.mean_
'取標準差'
#sc.scale_
x_train_std = sc.transform(x_train)
x_test_std=sc.transform(x_test)
'x 垂直'
x_std = np.vstack((x_train_std,x_test_std))
'y 鋪平'
y_combined = np.hstack((y_train, y_test))
'func'
logistic = LogisticRegression(C=1000,random_state=10)
'loda test data'
logistic.fit(x_train_std,y_train)
'機率測試'
logistic.predict_proba(x_test_std)
'上述 Def  '
plot_decision_regions(x_std,y_combined,classifier=logistic,test_idx=range(120,150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()
#'用data 1 2進行判別'


#import matplotlib.pyplot as plt
#import numpy as np
#
##iris = datasets.load_iris()
##'用 2 3列做為regression data'
##x=iris.data[:,[0,1,2,3]]
##y=iris.target
##x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
#
##print(x_train)
##print(y_train)
##print(x_test)
##print(y_test)
#
##用0,1去做分類
#
#dataset = np.array([
#((5.1,3.5,1.4,0.2), 0),
#((4.9,3.0,1.4,0.2), 0),
#((4.7,3.2,1.3,0.2), 0),
#((4.6,3.1,1.5,0.2), 0),
#((5.0,3.6,1.4,0.2), 0), #非線性分割點
#
#((5.6,2.9,3.6,1.3), 1),
#((6.7,3.1,4.4,1.4), 1),
#((5.6,3.0,4.5,1.5), 1),
#((5.8,2.7,4.1,1.0), 1),
#((6.2,2.2,4.5,1.5), 1)])
#
#print(dataset.shape)
#
##計算機率函數
#
#def sigmoid(z):
#    return 1 / (1 + np.exp(-z))
#    
##計算平均梯度
#
#def gradient(dataset, w):
#    g = np.zeros(len(w))
#    for x,y in dataset:
#        x = np.array(x)
#        error = sigmoid(w.T.dot(x))
#        g += (error - y) * x
#    return g / len(dataset)
#
##計算現在的權重的錯誤有多少
#
#def cost(dataset, w):
#    total_cost = 0
#    for x,y in dataset:
#        x = np.array(x)
#        error = sigmoid(w.T.dot(x))
#        total_cost += abs(y - error)
#    return total_cost
#
#def logistic(dataset): #演算法實作
#
#    w = np.zeros(4) #用0 + 0*x1 + 0*x2當作初始設定 
#
#    limit = 10 #更新1000次後停下
#
#    eta = 1 #更新幅度
#
#    costs = [] #紀錄每次更新權重後新的cost是多少
#
#    for i in range(limit):
#        current_cost = cost(dataset, w)
#        print ("current_cost=",current_cost)
#        costs.append(current_cost)
#        w = w - eta * gradient(dataset, w)
#        eta *= 0.9 #更新幅度，逐步遞減
#
#    #畫出cost的變化曲線
#    plt.plot(range(limit), costs)
#    plt.show()
#    print(w[0],w[1],w[2])
#    return w
##執行
#
#
#w = logistic(dataset)
##畫圖
#
#
#ps = [v[0] for v in dataset]

#fig = plt.figure()
#ax1 = fig.add_subplot(3,1,2)
#'scatter parameter def'
#'(inputData,inputData,s = nums of data,c= color,maker = data sign , label)'
#ax1.scatter([v[1] for v in ps[:5]], [v[2] for v in ps[:5]], s=10, c='b', marker="o", label='O')
#ax1.scatter([v[1] for v in ps[5:]], [v[2] for v in ps[5:]], s=10, c='r', marker="x", label='X')
#l = np.linspace(-1,5)
#a,b = -w[1]/w[2], -w[0]/w[2]
#ax1.plot(l, a*l + b, 'b-')
#'legend =  數組表 '
#plt.legend(loc='upper left');
#plt.show()
