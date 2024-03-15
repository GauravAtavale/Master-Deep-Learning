# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 07:07:15 2024

@author: gnath
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix, accuracy_score

titanic_df = pd.read_csv("C:/Key files- GNA/Personal_study/Deep learning from scratch/Titanic_test_train.csv")

titanic_df.shape

titanic_df.columns

titanic_df.head


titanic_df.info()

titanic_df['Embarked'].head()

titanic_df['Fare']= np.round(titanic_df['Fare'],2)
#Build A generic RF model -- quickly

titanic_df = titanic_df.reset_index()
titanic_df.shape

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    return df[indices_to_keep].astype(np.float64)

titanic_df=clean_dataset(titanic_df)
titanic_df.shape

#.values
X = titanic_df.iloc[:, :-1]
y = titanic_df.iloc[:, -1]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# print(X_train)
# print(y_train)
# print(X_test)
# print(y_test)


# # Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


"""------------------------------------------------------------DL from scratch ------------------------------------------------------------"""


# init_weights_n1 = [random.random() for x in range(X_train.shape[1])]
# init_weights_n2 = [random.random() for x in range(X_train.shape[1])]

# hidden_final_weights = [random.random() for x in range(2)]

# bias_wt = [random.random() for x in range(2)]
# Define the activation function
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
while (accuracy_benchmark<0.83):
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
    print(i,accuracy_score(y_train, result))
    scores.append(accuracy_score(y_train, result))
    if accuracy_score(y_train, result) > accuracy_benchmark:
        accuracy_benchmark = accuracy_score(y_train, result)
        best_W1 = W1
        best_W2 = W2
        best_W3 = W3
    
    # Backpropagation 
    
    # db2_temp = A2 - np.array(y_train).reshape((len(y_train),1))
    # db2 = np.sum(db2_temp, axis=0)
    
    # dW2 = A1.T @ db2_temp
    
    # db1_temp = db2_temp @ W2.T
    
    # db1_temp[A1 <= 0] = 0
    
    # db1 = np.sum(db1_temp, axis=0)
        
    # dW1 = X_train.T @ db1_temp
    
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

print("Best Accuracy:",accuracy_benchmark)


"""-------------------------------------------- EXPERIMENTING- Trying to run with holdout data for diferent iterations --------------------------------------------------"""
#Like batch run. Need to implement softmax for the final layer tomorrow

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
    
X_train_MASTER = X_train.copy()
    
i=0    
while (accuracy_benchmark<0.85):
    i=i+1
    
    if i< 1000:
        X_train = X_train_MASTER[0:245]
    elif i<2000:
        X_train = X_train_MASTER[0:490]
    elif i<3000:
        X_train = X_train_MASTER[0:735]        
    else:
        X_train = X_train_MASTER
    
    
    # forward propagation
    
    # for i in range(10):
        
    Z1 = np.matmul(X_train, W1) 
    A1 = sigmoid(Z1)
    Z2 = np.matmul(A1, W2)
    A2 = sigmoid(Z2)
    Z3 = np.matmul(A2, W3)
    A3 = sigmoid(Z3)
    
    # result=[]
    # for each in np.array(A3).reshape(980):
    #     result.append(1 if each>=0.5 else 0)
        
    result=[]
    Z1_dummy = np.matmul(X_train_MASTER, W1) 
    A1_dummy = sigmoid(Z1_dummy)
    Z2_dummy = np.matmul(A1_dummy, W2)
    A2_dummy = sigmoid(Z2_dummy)
    Z3_dummy = np.matmul(A2_dummy, W3)
    A3_dummy = sigmoid(Z3_dummy)
    
    
    for each in np.array(A3_dummy).reshape(980):
        result.append(1 if each>=0.5 else 0)        
        
    
    #loss = BinaryCrossEntropy(y_train, np.array(A3).reshape(980))
    print(i,accuracy_score(y_train, result))
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
    #db3_temp = A3 - np.array(y_train).reshape((len(y_train),1))
    db3_temp = A3 - np.array(y_train[0:A3.shape[0]]).reshape((len(y_train[0:A3.shape[0]]),1))
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

print("Best Accuracy:",accuracy_benchmark)

