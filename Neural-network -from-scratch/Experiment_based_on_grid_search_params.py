# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 19:37:02 2024

@author: gnath
"""


# New experiment where I tweak the learning rate based on how close is the actual result to what I have
# I increase the learning rate if I am going in the positive direction 

import numpy as np
my_seed = 100
np.random.seed(my_seed)
import pandas as pd
import os
import matplotlib.pyplot as plt
import random
import sys
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm
import pickle

sys.path.append("C:/Key files- GNA/Personal_study/Deep learning from scratch")

import titanic_data_split

X_train, y_train, X_test, y_test = titanic_data_split.get_test_train_data_titanic()

def sigmoid(x):
    return 1/(1+ np.exp((-1)*x))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def ReLU(x):
    return x * (x > 0)

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

# get the weights based on grid search params
l = [32,64,128,256,512,1024]
#l = [32,128]

hidden_neurons_tup = [(x,y) for x in l for y in l]

global_accuracy = 0
for each_hl_comb in tqdm(hidden_neurons_tup):

    hidden_layer_n = each_hl_comb[0]
    hidden_layer_v2_n = each_hl_comb[1]
    
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
    while i<100:
        i=i+1
        # forward propagation
        
        # for i in range(10):
            
        Z1 = np.matmul(X_train, W1)
        A1 = sigmoid(Z1)
        Z2 = np.matmul(A1, W2)
        A2 = sigmoid(Z2)
        Z3 = np.matmul(A2, W3)
        A3 = sigmoid(Z3)
        
        result=[]
        for each in np.array(A3).reshape(980):
            result.append(1 if each>=0.5 else 0)
        
        #loss = BinaryCrossEntropy(y_train, np.array(A3).reshape(980))
        print(i,"h1:", hidden_layer_n, "h2:", hidden_layer_v2_n,"Accu:", accuracy_score(y_train, result))
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
        # dW1 = X_train.T @ db1_temp
        
        # Two hidden layer
        db3_temp = A3 - np.array(y_train).reshape((len(y_train),1))
        db3 = np.sum(db3_temp, axis=0)
        dW3 = A2.T @ db3_temp
        
        db2_temp = db3_temp @ W3.T
        db2_temp[A2 <= 0] = 0
        db2 = np.sum(db2_temp, axis=0)
        dW2 = A1.T @ db3_temp
        
        db1_temp = db2_temp @ W2.T
        db1_temp[A1 <= 0] = 0
        db1 = np.sum(db1_temp, axis=0)
        
        dW1 = X_train.T @ db1_temp
        
        if i<10:
            learning_rate=0.001
        else:
            learning_rate=0.000001
        
        
        W1,W2,W3 = W1 - learning_rate*dW1, W2- learning_rate*dW2, W3- learning_rate*dW3
        #b1,b2,b3 = b1 + learning_rate*db1, b2+ learning_rate*db2, b3+ learning_rate*db3
        
    if accuracy_benchmark > global_accuracy:
        global_accuracy  = accuracy_benchmark
        global_W1 = best_W1
        global_W2 = best_W2
        global_W3 = best_W3
        global_h1 = hidden_layer_n
        global_h2 = hidden_layer_v2_n
        

print("Best Accuracy:",global_h1,global_h2, global_accuracy)
len(scores)
plt.plot(scores)



for epoch in tqdm(hidden_neurons_tup):
    print(epoch)

for i in tqdm(range(int(9e6))):
    pass

for i in tqdm(range(0, 100)):
    print(1)
              

# - - - - -


# The best result so far
# 32 128
print( global_h1 , global_h2)

os.chdir("C:/Key files- GNA/Personal_study/Deep learning from scratch")

best_allover_w1 = global_W1
best_allover_w2 = global_W2
best_allover_w3 = global_W3

pickle.dumps(best_allover_w1)
pickle.dumps(best_allover_w2)
pickle.dumps(best_allover_w3)


with open('best_allover_w1.pkl','wb') as f:
    pickle.dump(best_allover_w1, f)

with open('best_allover_w2.pkl','wb') as f:
    pickle.dump(best_allover_w2, f)

with open('best_allover_w3.pkl','wb') as f:
    pickle.dump(best_allover_w3, f)    


with open('best_allover_w1.pkl','rb') as f:
    x = pickle.load(f)



Z1 = np.matmul(X_train, global_W1)
A1 = sigmoid(Z1)
Z2 = np.matmul(A1, global_W2)
A2 = sigmoid(Z2)
Z3 = np.matmul(A2, global_W3)
A3 = sigmoid(Z3)



result=[]
for each in np.array(A3).reshape(980):
    result.append(1 if each>=0.5 else 0)

#loss = BinaryCrossEntropy(y_train, np.array(A3).reshape(980))
print(accuracy_score(y_train, result))




