# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 17:54:42 2022

@author: gnath
"""

#LazyProgrammer Coding practice !!!

"""Neural Network feed forward coding"""
import numpy as np
import matplotlib.pyplot as plt

#Basic Data Creation
class_n=500

x1=np.random.randn(class_n,2) +np.array([0,-2])
x2=np.random.randn(class_n,2) +np.array([2,2])
x3=np.random.randn(class_n,2) +np.array([-2,2])

X=np.concatenate([x1,x2,x3])

Y=np.array([0]*class_n+[1]*class_n+[2]*class_n)

plt.scatter(X[:,0],X[:,1],c=Y, s=100,alpha=0.5)
plt.show()

D=2
M=3
K=3

w1=np.random.randn(D,M)
b1=np.random.randn(M)
w2=np.random.randn(M,K)
b2=np.random.randn(M)

def forward_feed(X, w1,b1,w2,b2):
    Z=1/(1+np.exp(-X.dot(w1)-b1)) #Coding Logit
    A=Z.dot(w2)+b2    #Value of softmax
    exp_A=np.exp(A)
    Y=exp_A/exp_A.sum(axis=1,keepdims=True)
    return Y
    
def classification_rate(Y,P):
    correct=0
    for i in range(len(Y)):
        if Y[i]==P[i]:
            correct+=1
    return correct/len(Y)
            
probabilities= forward_feed(X, w1,b1,w2,b2)
Predictions=np.argmax(probabilities,axis=1)

print(classification_rate(Y,Predictions))



























