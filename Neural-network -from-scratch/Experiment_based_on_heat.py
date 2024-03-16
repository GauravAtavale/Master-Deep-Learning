# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 19:37:02 2024

@author: gnath
"""


# New experiment where I tweak the learning rate based on how close is the actual result to what I have
# I increase the learning rate if I am going in the positive direction 

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random
import sys
from sklearn.metrics import confusion_matrix, accuracy_score

sys.path.append("C:/Key files- GNA/Personal_study/Deep learning from scratch")

import titanic_data_split

X_train, y_train, X_test, y_test = titanic_data_split.get_test_train_data_titanic()

def sigmoid(x):
    return 1/(1+ np.exp((-1)*x))

# Define BCE
def BinaryCrossEntropy(train_true,p_prob):
    train_true_temp=train_true.copy().reset_index(drop=True)
    total_loss = 0 
    for each in range(len(p_prob)):
        total_loss = total_loss + ((-1)*((train_true_temp[each] * np.log(p_prob[each])) + ((1-train_true_temp[each])*np.log(1-p_prob[each]))))
    return total_loss/(len(p_prob))

def partial_differential_func(x):
    logisic=1/(1+np.exp(-x))
    differential=logisic*(1-logisic)
    return differential

#Backup Loss function
def BinaryCrossEntropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    term_0 = (1-y_true) * np.log(1-y_pred + 1e-7)
    term_1 = y_true * np.log(y_pred + 1e-7)
    return -np.mean(term_0+term_1, axis=0)


# Define initial weights 

hidden_layer_n = 32
hidden_layer_v2_n = 1024

W1 = np.random.randn(X_train.shape[1], hidden_layer_n )
# b1 = np.zeros(hidden_layer_n)
b1 = np.random.randn(hidden_layer_n)

#Add another layer with same number of neurons
W2 = np.random.randn(hidden_layer_n, hidden_layer_v2_n)
#b2 = np.zeros(hidden_layer_v2_n)
b2 = np.random.randn(hidden_layer_v2_n)

W3 = np.random.randn(hidden_layer_v2_n, 1)
# b3 = np.zeros((1, 1))
b3 = np.random.randn(1, 1)

scores = []
accuracy_benchmark = 0
#for i in range(100):
i=0    
learning_rate=0.000001

prev_accu = 0
#while (accuracy_benchmark<0.85):
    
    
for each_row in X_train:

    # forward propagation
    
    # for i in range(10):
        
    Z1 = np.matmul(each_row, W1)  + b1
    A1 = sigmoid(Z1)
    Z2 = np.matmul(A1, W2) + b2
    A2 = sigmoid(Z2)
    Z3 = np.matmul(A2, W3) + b3
    A3 = sigmoid(Z3)
    
    # result=[]
    # for each in np.array(A3).reshape(980):
    #     result.append(1 if each>=0.5 else 0)
        
    if A3[0] >0.5:
        result = 1
    else:
        result = 0
    
    #loss = BinaryCrossEntropy(y_train, np.array(A3).reshape(980))
    #print(i,accuracy_score(y_train, result) , learning_rate)
    scores.append(accuracy_score(y_train, result))
    if accuracy_score(y_train, result) > accuracy_benchmark:
        accuracy_benchmark = accuracy_score(y_train, result)
        best_W1 = W1
        best_W2 = W2
        best_W3 = W3
    
    # Backpropagation 
    
    # If it was just one hidden layer
    # db2_temp = A2 - np.array(y_train).reshape((len(y_train),1))
    # db2 = np.sum(db2_temp, axis=0)
    # dW2 = A1.T @ db2_temp
    # db1_temp = db2_temp @ W2.T
    # db1_temp[A1 <= 0] = 0
    # db1 = np.sum(db1_temp, axis=0)        
    # dW1 = each_row.T @ db1_temp
    
    # Two hidden layer
    
    y_train=y_train.copy().reset_index(drop=True)
    #np.array(y_train).reshape((len(y_train),1))
    db3_temp = A3 - y_train[i]
    db3 = np.sum(db3_temp, axis=0)
    dW3 = A2.T @ db3_temp
    
    db2_temp = db3_temp @ W3.T
    db2_temp[A2 <= 0] = 0
    db2 = np.sum(db2_temp, axis=0)
    dW2 = A1.T @ db3_temp
    
    db1_temp = db2_temp @ W2.T
    db1_temp[A1 <= 0] = 0
    db1 = np.sum(db1_temp, axis=0)
    
    dW1 = each_row.T @ db1_temp
    

    
    W1,W2,W3 = W1 - learning_rate*dW1, W2- learning_rate*dW2, W3- learning_rate*dW3
    #b1,b2,b3 = b1 + learning_rate*db1, b2+ learning_rate*db2, b3+ learning_rate*db3
    
    i=i+1

print(i,accuracy_score(y_train, scores) , learning_rate)

print("Best Accuracy:",accuracy_benchmark)
len(scores)
plt.plot(scores)



# delta_accu = accuracy_score(y_train, result) - prev_accu

# if delta_accu>0:
#     learning_rate = learning_rate * 10 
# else:
#     learning_rate = learning_rate / 10
# prev_accu = accuracy_score(y_train, result)


####----------------


prev_accu = 0.8

new_accu = 0.81

delta_accu = new_accu - prev_accu

if delta_accu>0:
    learning_rate = learning_rate * 10 
else:
    learning_rate = learning_rate / 10 




