# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 20:14:18 2022

@author: gnath
"""

#------------------------------------------------------------------------------------
"""Day- 2: Working with ecom data and predicting using the forward loop"""

#Data Pre-processing
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

os.chdir(r"C:\Key files- GNA\Indiana University\Personal Goals\Neural Network and DNN\LazyProgrammer- Udemy-Deep Learning")


def get_data():
    data=pd.read_csv('ecommerce_data.csv')
    
    x_needed_cols=['is_mobile', 'n_products_viewed', 'visit_duration','is_returning_visitor', 'time_of_day']
    
    X = data.loc[:, x_needed_cols]
    Y = data.loc[:, ['user_action']]
    
    dummy_day=pd.get_dummies(X['time_of_day'],prefix='time')
    X = pd.merge(
        left=X,
        right=dummy_day,
        left_index=True,
        right_index=True,
    )
    X.drop('time_of_day', axis=1, inplace=True)
    
    from sklearn.preprocessing import StandardScaler
    data_scale=X
    scalar = StandardScaler().fit(data_scale)
    X=scalar.transform(X)
    return X,Y

def get_binary_data(X,Y):
    Y2=Y.loc[Y['user_action']<=0,:]    
    X2=X.loc[Y['user_action']<=0,:]
    return X2,Y2

#Getting predictions

X,Y=get_data()

M=5
D=X.shape[1]
K=len(set(Y.iloc[:,0].unique()))


w1=np.random.randn(D,M)
b1=np.zeros(M)
w2=np.random.randn(M,K)
b2=np.zeros(K)

def softmax(a):
    expA=np.exp(a)
    row_sum=expA.sum(axis=1,keepdims=True)
    return expA/row_sum

def forward_feed_tanh(X,w1,b1,w2,b2):
    Z=np.tanh(X.dot(w1)+b1)
    A=Z.dot(w2)+b2
    Y=softmax(A)
    return Y

def classification_rate(Y,P):
    correct=0
    for i in range(len(Y)):
        if Y[i]==P[i]:
            correct+=1
    return correct/len(Y)

probabilities= forward_feed_tanh(X, w1,b1,w2,b2)
Predictions=np.argmax(probabilities,axis=1)

print(classification_rate(Y.iloc[:,0],Predictions))