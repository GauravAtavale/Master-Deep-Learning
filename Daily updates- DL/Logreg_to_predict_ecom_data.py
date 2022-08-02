# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 21:30:02 2022

@author: gnath
"""

"""Day- 5: Logistic regression to predict ecom data"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.utils import shuffle

def get_data():
    os.chdir(r"C:\Key files- GNA\Indiana University\Personal Goals\Neural Network and DNN\LazyProgrammer- Udemy-Deep Learning")
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

def y2indicator(y,K):
    N=len(y)
    ind=np.zeros((N,K))
    for i in range(N):
        ind[i,y[i]]=1
    return ind

X,Y = get_data()
X,Y = shuffle(X,Y)

Y=Y.astype(np.int32)
D=X.shape[1]
K=len(set(Y.iloc[:,0].unique()))

X_train=X[:-100]
Y_train=Y[:-100]
Y_train_ind=y2indicator(np.array(Y_train), K)

X_test=X[-100:]
Y_test=Y[-100:]
Y_test_ind=y2indicator(np.array(Y_test), K)

W=np.random.randn(D,K)
b=np.zeros(K)

def softmax(a):
    expA=np.exp(a)
    row_sum=expA.sum(axis=1,keepdims=True)
    return expA/row_sum

def forward(X,W,b):
    return softmax(X.dot(W)+b)

def predict(P_Y_given_X):
    return np.argmax(P_Y_given_X,axis=1)

def classification_rate(Y,P):
    correct=0
    for i in range(len(Y)):
        if Y[i]==P[i]:
            correct+=1
    return correct/len(Y)
    #return np.mean(Y==P)


def cross_entropy(T,pY):
    return -np.mean(T*np.log(pY))

train_cost=[]
test_cost=[]
learning_rate=0.001

for i in range(1000000):
    pYtrain=forward(X_train,W,b)
    pYtest=forward(X_test,W,b)
    
    ctrain=cross_entropy(Y_train_ind, pYtrain)
    ctest=cross_entropy(Y_test_ind, pYtest)
    train_cost.append(ctrain)
    test_cost.append(ctest)
    
    W-=learning_rate*X_train.T.dot(pYtrain-Y_train_ind)
    b-=learning_rate*(pYtrain-Y_train_ind).sum(axis=0)
    
    if i %10000==0:
        print (i,ctrain,ctest)
        
        
print("Final train classification rate",classification_rate(np.array(Y_train), predict(pYtrain)))
print("Final test classification rate",classification_rate(np.array(Y_test), predict(pYtest)))
        
plt.plot(train_cost)
plt.plot(test_cost) 
plt.show()
