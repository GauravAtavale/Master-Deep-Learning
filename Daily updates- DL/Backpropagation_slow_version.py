# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 23:12:57 2022

@author: gnath
"""

#------------------------------------------------------------------------------------
"""Day- 3: Coding Backpropagation along with Forward feed"""

import numpy as np
import matplotlib.pyplot as plt

def forward_feed(X, w1,b1,w2,b2):
    Z=1/(1+np.exp(-X.dot(w1)-b1))
    A=Z.dot(w2)+b2    #Value of softmax
    exp_A=np.exp(A)
    Y=exp_A/exp_A.sum(axis=1,keepdims=True)
    return Y,Z
    
def classification_rate(Y,P):
    correct=0
    for i in range(len(Y)):
        if Y[i]==P[i]:
            correct+=1
    return correct/len(Y)

def cost(T,Y):
    tot=T*np.log(Y)
    return tot.sum()


def derivative_w2(Z,T,Y):
    N,K=T.shape
    M=Z.shape[1]
    
    ret1=np.zeros((M,K))
    for n in range(N):
        for m in range(M):
            for k in range(K):
                ret1[m,k]+=(T[n,k]-Y[n,k])*Z[n,m]
    return ret1

def derivative_b2(T,Y):
    return (T-Y).sum(axis=0)

def derivative_w1(X,Z,T,Y,w2):
    N,D=X.shape
    M,K=w2.shape
    
    ret1=np.zeros((D,M))
    for n in range(N):
        for k in range(K):
            for m in range(M):
                for d in range(D):                   
                    ret1[d,m]+=(T[n,k]-Y[n,k])*w2[m,k]*Z[n,m]*(1-Z[n,m])*X[n,d]
    return ret1
    
def derivative_b1(T,Y,w2,Z):
    return ((T-Y).dot(w2.T)*Z*(1-Z)).sum(axis=0)
    

#--------------------------------------------Main --------------------------------------------

#Basic Data Creation
class_n=500

D=2
M=3
K=3

x1=np.random.randn(class_n,2) +np.array([0,-2])
x2=np.random.randn(class_n,2) +np.array([2,2])
x3=np.random.randn(class_n,2) +np.array([-2,2])

#X=np.concatenate([x1,x2,x3])
X=np.vstack([x1,x2,x3])

Y=np.array([0]*class_n+[1]*class_n+[2]*class_n)
N=len(Y)

T=np.zeros((N,K))
for i in range(N):
    T[i,Y[i]]=1
#plt.scatter(X[:,0],X[:,1],c=Y, s=100,alpha=0.5)
#plt.show()
    

w1=np.random.randn(D,M)
b1=np.random.randn(M)
w2=np.random.randn(M,K)
b2=np.random.randn(M)

learning_rate=10e-7
costs=[]
from tqdm import tqdm
for epoch in tqdm(range(10000)):
    output,hidden=forward_feed(X, w1, b1, w2, b2)
    if epoch % 1000:
        c=cost(T,output)
        P=np.argmax(output,axis=1)
        r=classification_rate(Y, P)
        print("Cost:",c,"Classification rate:",r)
        costs.append(c)
        
    w2+=learning_rate * derivative_w2(hidden,T,output)
    b2+=learning_rate * derivative_b2(T,output)
    w1+=learning_rate * derivative_w1(X,hidden,T,output,w2)
    b1+=learning_rate * derivative_b1(T,output,w2,hidden)
plt.plot(costs)
plt.show()
